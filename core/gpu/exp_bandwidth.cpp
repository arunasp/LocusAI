// Device-memory and host<->device bandwidth on this boot.
//
// memoryClockRate is unreliable under GPU-PV (doc/core/HARDWARE.md), so
// bandwidth is timed, not derived. Feeds the ingestion IO check in
// doc/core/HIERARCHY.md: loading is hidden when rho < (B_IO / B_GPU) * C.
//
// Each figure is sampled until the median of the first half of the samples
// agrees with the median of the second half within the spread of all of
// them. A run that never stabilises ends on the cap and says so.
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

__global__ void copy_kernel(const float4 *__restrict__ in,
                            float4 *__restrict__ out, size_t n)
{
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += stride)
        out[i] = in[i];
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

// Six samples is the least that gives each half a median distinct from its
// extremes.
static bool stable(const std::vector<double> &s)
{
    if (s.size() < 6)
        return false;
    size_t h = s.size() / 2;
    std::vector<double> a(s.begin(), s.begin() + h);
    std::vector<double> b(s.begin() + h, s.end());
    return std::fabs(median_of(a) - median_of(b)) <= mad_of(s);
}

// Safety cap only. A run that reaches it is reported as cap-terminated.
static const size_t CAP = 500;

template <class F>
static double measure(const char *name, F sample, bool *capped)
{
    std::vector<double> s;
    while (!stable(s) && s.size() < CAP)
        s.push_back(sample());
    bool hit = !stable(s);
    if (hit)
        *capped = true;
    double md = median_of(s);
    std::printf("%-20s %9.1f GB/s  n=%zu  mad=%.1f  stopped_on=%s\n",
                name, md, s.size(), mad_of(s), hit ? "cap" : "stable");
    return md;
}

int main()
{
    hipDeviceProp_t p;
    CHECK(hipGetDeviceProperties(&p, 0));

    // Far above every on-chip cache (80 MB Infinity Cache on gfx1100), and
    // small enough that pinned host memory of the same size is obtainable.
    size_t bytes = (p.totalGlobalMem / 32) & ~(size_t)15;
    size_t n4 = bytes / sizeof(float4);
    std::printf("device %s (%s), buffer %.1f MiB\n", p.name,
                p.gcnArchName, bytes / 1048576.0);

    float4 *d_in, *d_out;
    CHECK(hipMalloc(&d_in, bytes));
    CHECK(hipMalloc(&d_out, bytes));
    CHECK(hipMemset(d_in, 1, bytes));

    void *h_pin;
    CHECK(hipHostMalloc(&h_pin, bytes, hipHostMallocDefault));
    void *h_page = std::malloc(bytes);
    if (!h_page) {
        std::fprintf(stderr, "FAIL: pageable host allocation\n");
        return 1;
    }
    std::fill((char *)h_page, (char *)h_page + bytes, 1);
    std::fill((char *)h_pin, (char *)h_pin + bytes, 1);

    hipEvent_t t0, t1;
    CHECK(hipEventCreate(&t0));
    CHECK(hipEventCreate(&t1));

    int threads = 256;
    int blocks = p.multiProcessorCount * 32;

    auto timed = [&](auto op, double moved) {
        CHECK(hipEventRecord(t0));
        op();
        CHECK(hipEventRecord(t1));
        CHECK(hipEventSynchronize(t1));
        float ms = 0.0f;
        CHECK(hipEventElapsedTime(&ms, t0, t1));
        return moved / (ms * 1.0e6);
    };

    auto dev_copy = [&] {
        copy_kernel<<<blocks, threads>>>(d_in, d_out, n4);
        CHECK(hipGetLastError());
    };
    auto h2d_pin = [&] { CHECK(hipMemcpy(d_in, h_pin, bytes, hipMemcpyHostToDevice)); };
    auto h2d_page = [&] { CHECK(hipMemcpy(d_in, h_page, bytes, hipMemcpyHostToDevice)); };
    auto d2h_pin = [&] { CHECK(hipMemcpy(h_pin, d_out, bytes, hipMemcpyDeviceToHost)); };

    bool capped = false;
    // Copy reads and writes every byte once.
    double gpu = measure("device copy (r+w)", [&] { return timed(dev_copy, 2.0 * bytes); }, &capped);
    double h2d = measure("host->dev pinned", [&] { return timed(h2d_pin, (double)bytes); }, &capped);
    double h2d_p = measure("host->dev pageable", [&] { return timed(h2d_page, (double)bytes); }, &capped);
    double d2h = measure("dev->host pinned", [&] { return timed(d2h_pin, (double)bytes); }, &capped);

    std::printf("B_IO/B_GPU pinned   %.4f\n", h2d / gpu);
    std::printf("B_IO/B_GPU pageable %.4f\n", h2d_p / gpu);
    std::printf("d2h/h2d pinned      %.3f\n", d2h / h2d);

    CHECK(hipEventDestroy(t0));
    CHECK(hipEventDestroy(t1));
    CHECK(hipHostFree(h_pin));
    std::free(h_page);
    CHECK(hipFree(d_in));
    CHECK(hipFree(d_out));

    if (capped) {
        std::printf("RESULT: at least one figure ended on the cap -- not a measurement\n");
        return 2;
    }
    std::printf("PASS: bandwidth\n");
    return 0;
}
