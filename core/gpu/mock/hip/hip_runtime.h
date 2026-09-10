// Mock of exactly the HIP surface caps.cpp uses -- nothing more.
//
// PURPOSE: caps.cpp branches on what the runtime exposes (optional struct
// fields, attribute support, error codes) and on whether a launch covered
// the whole grid. Those branches decide what the probe REPORTS, so shipping
// them unexercised means shipping a report whose failure paths have never
// run. This mock runs the real caps.cpp source on the host, with each
// branch selectable by a -D knob.
//
// It is NOT a HIP emulator and must not grow into one. Device pointers are
// host pointers and a "launch" is a serial loop, which is exactly enough to
// verify caps.cpp's own logic and nothing about the GPU.
//
// Scenario knobs, all defaulting to the healthy case:
//   MOCK_DEVICE_COUNT      device count hipGetDeviceCount reports
//   MOCK_WARP_SIZE         warpSize the mock device claims
//   MOCK_LDS               sharedMemPerBlock the mock device claims
//   MOCK_OMIT_OPTIONAL     1 = struct lacks the newer fields entirely
//   MOCK_PARTIAL_LAUNCH    1 = last block never runs (a real fault shape)
//   MOCK_LIMIT_UNSUPPORTED 1 = hipDeviceGetLimit refuses the printf limit
//   MOCK_FUNCATTR_FAIL     1 = hipFuncGetAttributes returns not-supported
//   MOCK_PHYS_ATTR_FAIL    1 = PhysicalMultiProcessorCount unsupported

#ifndef LOCUS_MOCK_HIP_RUNTIME_H
#define LOCUS_MOCK_HIP_RUNTIME_H

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifndef MOCK_DEVICE_COUNT
#  define MOCK_DEVICE_COUNT 1
#endif
#ifndef MOCK_WARP_SIZE
#  define MOCK_WARP_SIZE 32
#endif
#ifndef MOCK_LDS
#  define MOCK_LDS 65536
#endif
#ifndef MOCK_OMIT_OPTIONAL
#  define MOCK_OMIT_OPTIONAL 0
#endif
#ifndef MOCK_PARTIAL_LAUNCH
#  define MOCK_PARTIAL_LAUNCH 0
#endif
#ifndef MOCK_LIMIT_UNSUPPORTED
#  define MOCK_LIMIT_UNSUPPORTED 0
#endif
#ifndef MOCK_FUNCATTR_FAIL
#  define MOCK_FUNCATTR_FAIL 0
#endif
#ifndef MOCK_PHYS_ATTR_FAIL
#  define MOCK_PHYS_ATTR_FAIL 0
#endif
#ifndef MOCK_COOP_UNSUPPORTED
#  define MOCK_COOP_UNSUPPORTED 0
#endif
#ifndef MOCK_COOP_REJECT
#  define MOCK_COOP_REJECT 0
#endif
#ifndef MOCK_OCC_FAIL
#  define MOCK_OCC_FAIL 0
#endif

#define __global__
#define __device__
#define __host__
// LDS becomes a plain static array. That is faithful ONLY because the
// kernels here have each thread touch its own slot and zero it first, so a
// serial thread sweep produces the same values a real block would. It does
// NOT model a barrier: __syncthreads is a no-op below, so any kernel whose
// RESULT depends on cross-thread ordering is outside what this mock can
// verify, and asserting such a result here would be asserting nothing.
#define __shared__ static
inline void __syncthreads() {}

// Error codes: the numeric values match HIP's, because the DISTINCTION
// between NotSupported and UnsupportedLimit is the finding caps.cpp reports.
typedef enum hipError_t {
    hipSuccess = 0,
    hipErrorInvalidValue = 1,
    hipErrorNotSupported = 801,
    hipErrorUnsupportedLimit = 802,
    hipErrorUnknown = 999
} hipError_t;

enum hipLimit_t {
    hipLimitStackSize = 0x0,
    hipLimitPrintfFifoSize = 0x01,
    hipLimitMallocHeapSize = 0x02,
    hipLimitRange
};

enum hipFuncCache_t {
    hipFuncCachePreferNone = 0,
    hipFuncCachePreferShared = 1,
    hipFuncCachePreferL1 = 2,
    hipFuncCachePreferEqual = 3
};

enum hipDeviceAttribute_t {
    hipDeviceAttributeMultiprocessorCount = 1,
    hipDeviceAttributePhysicalMultiProcessorCount = 2,
    hipDeviceAttributeCooperativeLaunch = 3,
    hipDeviceAttributeWallClockRate = 4
};

enum hipMemcpyKind {
    hipMemcpyHostToHost = 0,
    hipMemcpyHostToDevice = 1,
    hipMemcpyDeviceToHost = 2,
    hipMemcpyDeviceToDevice = 3
};

struct dim3 {
    unsigned int x, y, z;
    dim3(unsigned int x_ = 1, unsigned int y_ = 1, unsigned int z_ = 1)
        : x(x_), y(y_), z(z_) {}
};

inline dim3 blockIdx(0, 0, 0);
inline dim3 threadIdx(0, 0, 0);
inline dim3 blockDim(1, 1, 1);
inline dim3 gridDim(1, 1, 1);

struct hipDeviceProp_t {
    char name[256];
    char gcnArchName[256];
    size_t totalGlobalMem;
    size_t sharedMemPerBlock;
    int regsPerBlock;
    int warpSize;
    int multiProcessorCount;
    int l2CacheSize;
    int memoryBusWidth;
    int memoryClockRate;
    int concurrentKernels;
    int maxThreadsPerMultiProcessor;
#if !MOCK_OMIT_OPTIONAL
    int persistingL2CacheMaxSize;
    int accessPolicyMaxWindowSize;
    size_t maxSharedMemoryPerMultiProcessor;
    int regsPerMultiprocessor;
    size_t reservedSharedMemPerBlock;
    int globalL1CacheSupported;
    int kernelExecTimeoutEnabled;
    int cooperativeLaunch;
#endif
};

struct hipFuncAttributes {
    int binaryVersion;
    int cacheModeCA;
    size_t constSizeBytes;
    size_t localSizeBytes;
    int maxDynamicSharedSizeBytes;
    int maxThreadsPerBlock;
    int numRegs;
    int preferredShmemCarveout;
    int ptxVersion;
    size_t sharedSizeBytes;
};

inline hipError_t hipGetDeviceCount(int *n) {
    *n = MOCK_DEVICE_COUNT;
    return hipSuccess;
}

inline hipError_t hipGetDeviceProperties(hipDeviceProp_t *p, int) {
    std::memset(p, 0, sizeof *p);
    std::snprintf(p->name, sizeof p->name, "MOCK device");
    std::snprintf(p->gcnArchName, sizeof p->gcnArchName, "mockarch");
    p->totalGlobalMem = (size_t)20 * 1024 * 1024 * 1024;
    p->sharedMemPerBlock = (size_t)(MOCK_LDS);
    p->regsPerBlock = 65536;
    p->warpSize = MOCK_WARP_SIZE;
    p->multiProcessorCount = 42;
    p->l2CacheSize = 6 * 1024 * 1024;
    p->memoryBusWidth = 320;
    p->memoryClockRate = 1250000;
    p->concurrentKernels = 1;
    p->maxThreadsPerMultiProcessor = 2048;
#if !MOCK_OMIT_OPTIONAL
    p->persistingL2CacheMaxSize = 0;
    p->accessPolicyMaxWindowSize = 0;
    p->maxSharedMemoryPerMultiProcessor = 65536;
    p->regsPerMultiprocessor = 65536;
    p->reservedSharedMemPerBlock = 0;
    p->globalL1CacheSupported = 0;
    p->kernelExecTimeoutEnabled = 0;
    p->cooperativeLaunch = 1;
#endif
    return hipSuccess;
}

inline hipError_t hipDeviceGetAttribute(int *v, hipDeviceAttribute_t a, int) {
    switch (a) {
        case hipDeviceAttributeMultiprocessorCount:
            *v = 42;
            return hipSuccess;
        case hipDeviceAttributeCooperativeLaunch:
            *v = MOCK_COOP_UNSUPPORTED ? 0 : 1;
            return hipSuccess;
        case hipDeviceAttributeWallClockRate:
            *v = 100000;
            return hipSuccess;
        case hipDeviceAttributePhysicalMultiProcessorCount:
#if MOCK_PHYS_ATTR_FAIL
            return hipErrorNotSupported;
#else
            *v = 84;
            return hipSuccess;
#endif
        default:
            return hipErrorInvalidValue;
    }
}

inline hipError_t hipFuncGetAttributes(hipFuncAttributes *fa, const void *) {
#if MOCK_FUNCATTR_FAIL
    (void)fa;
    return hipErrorNotSupported;
#else
    std::memset(fa, 0, sizeof *fa);
    fa->numRegs = 24;
    fa->localSizeBytes = 0;
    fa->sharedSizeBytes = 0;
    fa->maxThreadsPerBlock = 1024;
    return hipSuccess;
#endif
}

// The real one is documented to accept only two of the three enum members.
// The mock reproduces that observable behaviour rather than succeeding for
// everything -- a fake that always succeeds tests nothing.
inline hipError_t hipDeviceGetLimit(size_t *v, enum hipLimit_t l) {
    switch (l) {
        case hipLimitStackSize: *v = 1024; return hipSuccess;
        case hipLimitMallocHeapSize: *v = 8u * 1024 * 1024; return hipSuccess;
        case hipLimitPrintfFifoSize:
#if MOCK_LIMIT_UNSUPPORTED
            *v = 0;
            return hipErrorUnsupportedLimit;
#else
            *v = 4096;
            return hipSuccess;
#endif
        default: *v = 0; return hipErrorUnsupportedLimit;
    }
}

inline hipError_t hipDeviceSetLimit(enum hipLimit_t, size_t) {
    return hipSuccess;
}

inline hipError_t hipDeviceSetCacheConfig(hipFuncCache_t) {
    return hipErrorNotSupported;  // matches the documented AMD behaviour
}

inline hipError_t hipMalloc(void **p, size_t n) {
    *p = std::malloc(n);
    return *p ? hipSuccess : hipErrorUnknown;
}
template <typename T>
inline hipError_t hipMalloc(T **p, size_t n) {
    *p = (T *)std::malloc(n);
    return *p ? hipSuccess : hipErrorUnknown;
}
inline hipError_t hipFree(void *p) {
    std::free(p);
    return hipSuccess;
}
inline hipError_t hipMemcpy(void *d, const void *s, size_t n, hipMemcpyKind) {
    std::memcpy(d, s, n);
    return hipSuccess;
}
inline hipError_t hipDeviceSynchronize() { return hipSuccess; }
inline hipError_t hipGetLastError() { return hipSuccess; }
inline const char *hipGetErrorString(hipError_t) { return "mock"; }

// A "launch" is a serial sweep of the grid. MOCK_PARTIAL_LAUNCH drops the
// last block, which is the fault shape a spot-check verifier misses and the
// reason caps.cpp verifies every element.
template <typename F, typename... Args>
inline void hipLaunchKernelGGL(F kernel, dim3 grid, dim3 block, size_t,
                               void *, Args... args) {
    gridDim = grid;
    blockDim = block;
    unsigned int nb = grid.x;
#if MOCK_PARTIAL_LAUNCH
    if (nb > 0) nb -= 1;
#endif
    for (unsigned int b = 0; b < nb; ++b) {
        blockIdx = dim3(b, 0, 0);
        for (unsigned int t = 0; t < block.x; ++t) {
            threadIdx = dim3(t, 0, 0);
            kernel(args...);
        }
    }
}

inline hipError_t hipOccupancyMaxActiveBlocksPerMultiprocessor(
    int *blocks, const void *, int, size_t) {
#if MOCK_OCC_FAIL
    *blocks = 0;
    return hipErrorNotSupported;
#else
    *blocks = 2;
    return hipSuccess;
#endif
}
template <typename T>
inline hipError_t hipOccupancyMaxActiveBlocksPerMultiprocessor(int *blocks, T f,
                                                               int bs,
                                                               size_t dyn) {
    return hipOccupancyMaxActiveBlocksPerMultiprocessor(
        blocks, reinterpret_cast<const void *>(f), bs, dyn);
}

// Supports exactly ONE kernel signature -- void(float*, int, int) -- which is
// the only cooperative kernel in this project. Decoding a void** arg array
// generically is not possible in a host mock, and pretending otherwise would
// mean a fake that succeeds without running anything.
inline hipError_t hipLaunchCooperativeKernel(const void *f, dim3 grid,
                                             dim3 block, void **args, size_t,
                                             void *) {
#if MOCK_COOP_REJECT
    (void)f; (void)grid; (void)block; (void)args;
    return hipErrorNotSupported;
#else
    typedef void (*kfn)(float *, int, int);
    kfn k = reinterpret_cast<kfn>(const_cast<void *>(f));
    float *a0 = *reinterpret_cast<float **>(args[0]);
    int a1 = *reinterpret_cast<int *>(args[1]);
    int a2 = *reinterpret_cast<int *>(args[2]);
    hipLaunchKernelGGL(k, grid, block, 0, nullptr, a0, a1, a2);
    return hipSuccess;
#endif
}

#endif  // LOCUS_MOCK_HIP_RUNTIME_H
