// Toolchain probe: proves the build system can reach the device.
//
// Not a benchmark and not substrate code. It answers one question -- can
// core/ compile a HIP translation unit, launch a kernel, and read back a
// verified result on this machine -- so that when a real kernel is written,
// a failure is unambiguously the kernel's and not the toolchain's.
//
// It also prints the device properties the substrate design depends on:
// warp size (the natural competition granularity), LDS per block, and
// compute-unit count. These come from the HIP runtime rather than from
// rocminfo, because what matters is what the code we compile can see.
//
// Deliberately does no timing. A timing harness measures something; this
// establishes that there is something to measure.

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#define HIP_CHECK(expr)                                                    \
    do {                                                                   \
        hipError_t err_ = (expr);                                          \
        if (err_ != hipSuccess) {                                          \
            std::fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__,    \
                         #expr, hipGetErrorString(err_));                  \
            return 1;                                                      \
        }                                                                  \
    } while (0)

// Chosen so a wrong answer is obvious rather than plausible: every element
// has a distinct expected value, so a partially launched grid or a stale
// buffer shows up as a specific index mismatch instead of a number that
// looks roughly right.
__global__ void probe_kernel(const float *a, const float *b, float *out,
                             int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] * 2.0f + b[i];
    }
}

int main(void) {
    int devices = 0;
    HIP_CHECK(hipGetDeviceCount(&devices));
    if (devices == 0) {
        std::fprintf(stderr, "gpu probe: no HIP device visible\n");
        return 1;
    }

    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    std::printf("device        : %s (%s)\n", prop.name, prop.gcnArchName);
    std::printf("compute units : %d\n", prop.multiProcessorCount);
    std::printf("warp size     : %d\n", prop.warpSize);
    std::printf("LDS per block : %zu bytes\n",
                (size_t)prop.sharedMemPerBlock);
    std::printf("global memory : %.2f GiB\n",
                (double)prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    std::printf("clock         : %d kHz\n", prop.clockRate);

    const int n = 1 << 20;
    std::vector<float> a(n), b(n), out(n, -1.0f);
    for (int i = 0; i < n; ++i) {
        a[i] = (float)i;
        b[i] = (float)(n - i);
    }

    float *da = nullptr, *db = nullptr, *dout = nullptr;
    HIP_CHECK(hipMalloc(&da, n * sizeof(float)));
    HIP_CHECK(hipMalloc(&db, n * sizeof(float)));
    HIP_CHECK(hipMalloc(&dout, n * sizeof(float)));
    HIP_CHECK(hipMemcpy(da, a.data(), n * sizeof(float),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(db, b.data(), n * sizeof(float),
                        hipMemcpyHostToDevice));

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    hipLaunchKernelGGL(probe_kernel, dim3(blocks), dim3(threads), 0, 0,
                       da, db, dout, n);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(out.data(), dout, n * sizeof(float),
                        hipMemcpyDeviceToHost));

    // Verify every element, not a sample. A partial launch is exactly the
    // failure a spot check misses.
    int bad = 0;
    for (int i = 0; i < n; ++i) {
        float want = a[i] * 2.0f + b[i];
        if (out[i] != want) {
            if (bad == 0) {
                std::fprintf(stderr,
                             "gpu probe: mismatch at %d: got %f want %f\n",
                             i, out[i], want);
            }
            ++bad;
        }
    }

    HIP_CHECK(hipFree(da));
    HIP_CHECK(hipFree(db));
    HIP_CHECK(hipFree(dout));

    if (bad != 0) {
        std::fprintf(stderr, "gpu probe: %d/%d elements wrong\n", bad, n);
        return 1;
    }
    std::printf("kernel        : %d elements verified\n", n);
    std::printf("PASS: gpu\n");
    return 0;
}
