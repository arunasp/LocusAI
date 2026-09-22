// What each precision actually costs on THIS card.
//
// The learner and the field are written in double. On RDNA3 a double
// costs twice the bytes of a float and its arithmetic runs at a lower
// rate, so precision is a performance decision, not a style one. This
// measures both halves of that decision rather than quoting a spec:
//
//   arithmetic: fused multiply-adds held in registers (no memory)
//   streaming:  a sum over a large array (bandwidth bound)
//
// Each figure is sampled until the median of the first half of the
// samples agrees with the median of the second half within the spread
// of all of them, the rule exp_bandwidth.cpp already uses; a figure
// that never stabilises is reported as unstable rather than as a
// result.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(x)                                                         \
    do {                                                                 \
        hipError_t e_ = (x);                                             \
        if (e_ != hipSuccess) {                                          \
            std::fprintf(stderr, "FAIL: %s: %s\n", #x,                   \
                         hipGetErrorString(e_));                         \
            std::exit(1);                                                \
        }                                                                \
    } while (0)

static const int ITERS = 512;      // FMAs per thread per launch
static const int CHAIN = 8;        // independent chains, to fill the pipe

template <typename T>
__global__ void fma_kernel(T *out, T seed)
{
    T a[CHAIN], b = (T)1.0000001, c = (T)0.9999999;
    for (int k = 0; k < CHAIN; ++k)
        a[k] = seed + (T)k;
    for (int i = 0; i < ITERS; ++i)
        for (int k = 0; k < CHAIN; ++k)
            a[k] = a[k] * b + c;
    T s = (T)0;
    for (int k = 0; k < CHAIN; ++k)
        s += a[k];
    // One real store per block. A guard the compiler can PROVE false
    // (threadIdx.x == 1024 in a 256-thread block) lets it delete the
    // whole loop as dead -- which is exactly what happened the first
    // time this was run: fp32 and fp64 both reported 46594.1 GFLOP/s,
    // an empty launch timed twice.
    if (threadIdx.x == 0)
        out[blockIdx.x] = s;
}

template <typename T>
__global__ void stream_kernel(const T *in, T *out, long long n)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    T s = (T)0;
    for (; i < n; i += stride)
        s += in[i];
    if (threadIdx.x == 0)
        out[blockIdx.x] = s;      // see fma_kernel: the store must be real
}

static bool stable(std::vector<double> &v, double *med)
{
    if (v.size() < 6)
        return false;
    std::vector<double> a(v.begin(), v.begin() + v.size() / 2);
    std::vector<double> b(v.begin() + v.size() / 2, v.end());
    std::vector<double> all(v);
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    std::sort(all.begin(), all.end());
    double ma = a[a.size() / 2], mb = b[b.size() / 2];
    double spread = all.back() - all.front();
    *med = all[all.size() / 2];
    return std::fabs(ma - mb) <= spread;
}

template <typename T>
static double fma_rate(int blocks, int threads, bool *ok)
{
    T *out;
    CHECK(hipMalloc(&out, sizeof(T) * blocks));
    fma_kernel<T><<<blocks, threads>>>(out, (T)1.0);
    CHECK(hipDeviceSynchronize());
    std::vector<double> s;
    double med = 0;
    for (int i = 0; i < 40; ++i) {
        hipEvent_t a, b;
        CHECK(hipEventCreate(&a));
        CHECK(hipEventCreate(&b));
        CHECK(hipEventRecord(a));
        fma_kernel<T><<<blocks, threads>>>(out, (T)1.0);
        CHECK(hipEventRecord(b));
        CHECK(hipEventSynchronize(b));
        float ms = 0;
        CHECK(hipEventElapsedTime(&ms, a, b));
        CHECK(hipEventDestroy(a));
        CHECK(hipEventDestroy(b));
        double flops = 2.0 * (double)blocks * threads * ITERS * CHAIN;
        s.push_back(flops / (ms * 1e-3) / 1e9);
        if (stable(s, &med)) {
            *ok = true;
            CHECK(hipFree(out));
            return med;
        }
    }
    *ok = false;
    CHECK(hipFree(out));
    return med;
}

template <typename T>
static double stream_rate(long long n, int blocks, int threads, bool *ok)
{
    T *in, *out;
    CHECK(hipMalloc(&in, sizeof(T) * n));
    CHECK(hipMalloc(&out, sizeof(T) * blocks));
    CHECK(hipMemset(in, 0, sizeof(T) * n));
    stream_kernel<T><<<blocks, threads>>>(in, out, n);
    CHECK(hipDeviceSynchronize());
    std::vector<double> s;
    double med = 0;
    for (int i = 0; i < 40; ++i) {
        hipEvent_t a, b;
        CHECK(hipEventCreate(&a));
        CHECK(hipEventCreate(&b));
        CHECK(hipEventRecord(a));
        stream_kernel<T><<<blocks, threads>>>(in, out, n);
        CHECK(hipEventRecord(b));
        CHECK(hipEventSynchronize(b));
        float ms = 0;
        CHECK(hipEventElapsedTime(&ms, a, b));
        CHECK(hipEventDestroy(a));
        CHECK(hipEventDestroy(b));
        s.push_back((double)n * sizeof(T) / (ms * 1e-3) / 1e9);
        if (stable(s, &med)) {
            *ok = true;
            CHECK(hipFree(in));
            CHECK(hipFree(out));
            return med;
        }
    }
    *ok = false;
    CHECK(hipFree(in));
    CHECK(hipFree(out));
    return med;
}

int main()
{
    hipDeviceProp_t p;
    CHECK(hipGetDeviceProperties(&p, 0));
    int blocks = p.multiProcessorCount * 8, threads = 256;
    std::printf("%s, %d CUs, %d blocks x %d threads\n", p.gcnArchName,
                p.multiProcessorCount, blocks, threads);

    bool o32 = false, o64 = false;
    double f32 = fma_rate<float>(blocks, threads, &o32);
    double f64 = fma_rate<double>(blocks, threads, &o64);
    std::printf("arithmetic  fp32 %8.1f GFLOP/s%s\n", f32,
                o32 ? "" : "  UNSTABLE");
    std::printf("arithmetic  fp64 %8.1f GFLOP/s%s\n", f64,
                o64 ? "" : "  UNSTABLE");
    if (f64 > 0)
        std::printf("            fp32 is %.1fx fp64\n", f32 / f64);

    long long n = 1LL << 26;           // 64 Mi elements
    bool s32 = false, s64 = false;
    double b32 = stream_rate<float>(n, blocks, threads, &s32);
    double b64 = stream_rate<double>(n, blocks, threads, &s64);
    std::printf("streaming   fp32 %8.1f GB/s%s\n", b32,
                s32 ? "" : "  UNSTABLE");
    std::printf("streaming   fp64 %8.1f GB/s%s\n", b64,
                s64 ? "" : "  UNSTABLE");
    std::printf("            per VALUE, fp32 moves half the bytes: "
                "%.2f vs %.2f Gvalue/s\n", b32 / 4.0, b64 / 8.0);
    return (o32 && o64 && s32 && s64) ? 0 : 1;
}
