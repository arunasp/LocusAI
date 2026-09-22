// GPU memory hierarchy: read bandwidth over working sets from 256 KiB to
// 1 GiB, so L2 (6 MiB), Infinity Cache (80 MB) and VRAM show as plateaus,
// plus LDS read bandwidth measured separately.
//
// Each figure is sampled until the median of the first half of the samples
// agrees with the median of the second half within the spread of all of
// them. A figure that never stabilises ends on the cap and says so.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
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

static const int BD = 256;

// Reads the working set `passes` times; one float per thread is written
// so the reads cannot be removed.
__global__ void read_kernel(const float4 *__restrict__ in, size_t n4,
                            int passes, float *__restrict__ out)
{
    size_t tid = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    float acc = 0.0f;
    for (int p = 0; p < passes; ++p)
        for (size_t i = tid; i < n4; i += stride) {
            float4 v = in[i];
            acc += v.x + v.y + v.z + v.w;
        }
    out[tid] = acc;
}

// Non-constant fill: a uniform buffer may be served compressed, which
// would read faster than the memory it stands for.
__global__ void fill_kernel(float4 *buf, size_t n4)
{
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n4; i += stride) {
        uint32_t h = (uint32_t)i * 2654435761u;
        buf[i] = make_float4(h & 1023, (h >> 10) & 1023, (h >> 20) & 1023, i & 1023);
    }
}

// Each block fills 64 KiB of LDS once, then reads it `passes` times.
__global__ void lds_kernel(int passes, float *__restrict__ out)
{
    __shared__ float4 sh[4096];
    for (int i = threadIdx.x; i < 4096; i += blockDim.x)
        sh[i] = make_float4(i, i + 1, i + 2, i + 3);
    __syncthreads();
    float acc = 0.0f;
    for (int p = 0; p < passes; ++p)
        for (int i = threadIdx.x; i < 4096; i += blockDim.x) {
            float4 v = sh[(i + p) & 4095];
            acc += v.x + v.y + v.z + v.w;
        }
    out[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

static double median_of(std::vector<double> v)
{
    std::sort(v.begin(), v.end());
    size_t m = v.size() / 2;
    return v.size() % 2 ? v[m] : 0.5 * (v[m - 1] + v[m]);
}

static double mad_of(const std::vector<double> &v)
{
    double md = median_of(v);
    std::vector<double> d;
    for (double x : v)
        d.push_back(std::fabs(x - md));
    return median_of(d);
}

static bool stable(const std::vector<double> &s)
{
    if (s.size() < 6)
        return false;
    size_t h = s.size() / 2;
    std::vector<double> a(s.begin(), s.begin() + h), b(s.begin() + h, s.end());
    return std::fabs(median_of(a) - median_of(b)) <= mad_of(s);
}

static const size_t CAP = 300;
static int capped_total = 0;

template <class F>
static void measure(const char *label, F sample)
{
    std::vector<double> s;
    while (!stable(s) && s.size() < CAP)
        s.push_back(sample());
    bool hit = !stable(s);
    capped_total += hit;
    std::printf("%-22s %9.1f GB/s  n=%-3zu mad=%.1f  stopped_on=%s\n", label,
                median_of(s), s.size(), mad_of(s), hit ? "cap" : "stable");
    std::fflush(stdout);
}

int main()
{
    hipDeviceProp_t p;
    CHECK(hipGetDeviceProperties(&p, 0));
    std::printf("device %s (%s), L2 %d KiB reported\n", p.name, p.gcnArchName,
                p.l2CacheSize / 1024);

    const size_t max_bytes = (size_t)1 << 30;
    int blocks = p.multiProcessorCount * 32;
    float4 *buf;
    float *out;
    CHECK(hipMalloc(&buf, max_bytes));
    const char *fill = std::getenv("GPU_CACHE_FILL");
    bool zero = fill && std::string(fill) == "zero";
    if (zero) {
        CHECK(hipMemset(buf, 0, max_bytes));
    } else {
        fill_kernel<<<blocks, BD>>>(buf, max_bytes / sizeof(float4));
        CHECK(hipGetLastError());
    }
    CHECK(hipDeviceSynchronize());
    std::printf("fill: %s\n", zero ? "zero (hipMemset)" : "non-constant");
    CHECK(hipMalloc(&out, (size_t)blocks * BD * sizeof(float)));

    hipEvent_t t0, t1;
    CHECK(hipEventCreate(&t0));
    CHECK(hipEventCreate(&t1));
    auto timed = [&](auto launch, double bytes) {
        CHECK(hipEventRecord(t0));
        launch();
        CHECK(hipEventRecord(t1));
        CHECK(hipEventSynchronize(t1));
        float ms = 0.0f;
        CHECK(hipEventElapsedTime(&ms, t0, t1));
        return bytes / (ms * 1.0e6);
    };

    std::printf("\n== read sweep (whole GPU) ==\n");
    for (size_t bytes = (size_t)256 << 10; bytes <= max_bytes; bytes <<= 1) {
        size_t n4 = bytes / sizeof(float4);
        // At least 2 GiB of reads per sample, so a sample is long enough
        // to time whatever the working set.
        int passes = (int)std::max<size_t>(1, ((size_t)2 << 30) / bytes);
        auto launch = [&] {
            read_kernel<<<blocks, BD>>>(buf, n4, passes, out);
            CHECK(hipGetLastError());
        };
        launch();  // warm the caches for this working set
        CHECK(hipDeviceSynchronize());
        char label[64];
        if (bytes < ((size_t)1 << 20))
            std::snprintf(label, sizeof label, "read %zu KiB", bytes >> 10);
        else
            std::snprintf(label, sizeof label, "read %zu MiB", bytes >> 20);
        measure(label, [&] { return timed(launch, (double)bytes * passes); });
    }

    std::printf("\n== LDS (whole GPU, 64 KiB per block) ==\n");
    {
        const int passes = 2000;
        double bytes = (double)blocks * 4096 * sizeof(float4) * passes;
        auto launch = [&] {
            lds_kernel<<<blocks, BD>>>(passes, out);
            CHECK(hipGetLastError());
        };
        launch();
        CHECK(hipDeviceSynchronize());
        measure("LDS read", [&] { return timed(launch, bytes); });
    }

    CHECK(hipEventDestroy(t0));
    CHECK(hipEventDestroy(t1));
    CHECK(hipFree(buf));
    CHECK(hipFree(out));

    if (capped_total) {
        std::printf("RESULT: %d figure(s) ended on the cap -- not a measurement\n",
                    capped_total);
        return 2;
    }
    std::printf("PASS: gpu cache\n");
    return 0;
}
