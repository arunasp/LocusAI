// Device capability and memory-hierarchy probe.
//
// probe.cpp answers "can this toolchain reach the device at all". This file
// answers the next question: WHAT does the device expose, and which of the
// memory hierarchy is under software control. Every value here is read from
// the HIP runtime rather than from rocminfo, because what matters for kernel
// design is what the compiled code can see.
//
// Deliberately does no timing, for the same reason probe.cpp does not: a
// timing harness measures something, this establishes what there is to
// measure. Cache-residency measurement is a separate artifact.
//
// Every claim it prints is either a read value or an assertion that can
// FAIL. Assertions are invariants of a working probe (a device exists, a
// wavefront is a sane width, a launch wrote every element), never
// device-specific constants -- this must not hardcode gfx1100, because the
// same source has to run on the native-boot /dev/kfd path and report
// whatever is there.

#include <hip/hip_runtime.h>

#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <string>
#include <sys/stat.h>
#include <type_traits>
#include <utility>  // std::declval -- libstdc++ leaks it via
                    // <type_traits>, libc++ does not; hipcc may use either
#include <vector>

// ---------------------------------------------------------------------------
// Non-temporal access: ONE injection point.
//
// These are clang builtins. hipcc is a clang++ driver so the real build
// should take the first branch, but a host-compiler build (or an older
// clang) takes the fallback. Both branches are overridable from the command
// line so a test can exercise either without needing the other compiler
// installed -- the alternative was contorting the test around whichever
// compiler happened to be present.
// ---------------------------------------------------------------------------
#ifndef LOCUS_HAS_NT
#  if defined(__has_builtin)
#    define LOCUS_HAS_NT \
        (__has_builtin(__builtin_nontemporal_store) && \
         __has_builtin(__builtin_nontemporal_load))
#  else
#    define LOCUS_HAS_NT 0
#  endif
#endif

#ifndef LOCUS_NT_LOAD
#  if LOCUS_HAS_NT
#    define LOCUS_NT_LOAD(p) __builtin_nontemporal_load(p)
#  else
#    define LOCUS_NT_LOAD(p) (*(p))
#  endif
#endif

#ifndef LOCUS_NT_STORE
#  if LOCUS_HAS_NT
#    define LOCUS_NT_STORE(v, p) __builtin_nontemporal_store((v), (p))
#  else
#    define LOCUS_NT_STORE(v, p) (*(p) = (v))
#  endif
#endif

// ---------------------------------------------------------------------------
// Optional-field detection.
//
// hipDeviceProp_t grows between ROCm releases. Referencing a field directly
// makes the probe fail to COMPILE on an older runtime, which is the wrong
// failure: the probe's job is to report that the runtime does not expose the
// field. FIELD() gives a present/absent branch instead, and both branches
// are exercised by the mock suite.
// ---------------------------------------------------------------------------
template <typename...>
using locus_void_t = void;

#define LOCUS_DETECT_FIELD(field)                                            \
    template <typename T, typename = void>                                   \
    struct locus_has_##field : std::false_type {};                           \
    template <typename T>                                                    \
    struct locus_has_##field<                                                \
        T, locus_void_t<decltype(std::declval<const T &>().field)>>          \
        : std::true_type {};                                                 \
    template <typename T>                                                    \
    long long locus_get_##field(const T &p, std::true_type) {                \
        return (long long)p.field;                                           \
    }                                                                        \
    template <typename T>                                                    \
    long long locus_get_##field(const T &, std::false_type) {                \
        return -1;                                                           \
    }                                                                        \
    template <typename T>                                                    \
    long long locus_field_##field(const T &p) {                              \
        return locus_get_##field(p, locus_has_##field<T>{});                  \
    }

LOCUS_DETECT_FIELD(persistingL2CacheMaxSize)
LOCUS_DETECT_FIELD(accessPolicyMaxWindowSize)
LOCUS_DETECT_FIELD(maxSharedMemoryPerMultiProcessor)
LOCUS_DETECT_FIELD(regsPerMultiprocessor)
LOCUS_DETECT_FIELD(reservedSharedMemPerBlock)
LOCUS_DETECT_FIELD(globalL1CacheSupported)
LOCUS_DETECT_FIELD(kernelExecTimeoutEnabled)
LOCUS_DETECT_FIELD(cooperativeLaunch)

// ---------------------------------------------------------------------------
// Reporting. Everything goes to stdout AND to a log file, because the caller
// reads the log rather than a pasted terminal.
// ---------------------------------------------------------------------------
static std::FILE *g_log = nullptr;
static int g_pass = 0;
static int g_fail = 0;

static void out(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
static void out(const char *fmt, ...) {
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    std::fputs(buf, stdout);
    if (g_log) std::fputs(buf, g_log);
}

// A check that cannot fail is decoration, so each one names the condition it
// would fail on.
static void check(bool ok, const char *what, const char *detail) {
    if (ok) {
        ++g_pass;
        out("  [ok]   %-38s %s\n", what, detail);
    } else {
        ++g_fail;
        out("  [FAIL] %-38s %s\n", what, detail);
    }
}

static const char *err_name(hipError_t e) {
    // hipGetErrorString is the runtime's own text; reporting the enum too
    // matters because the DISTINCTION between "not supported" and
    // "unsupported limit" is the whole finding here.
    switch ((int)e) {
        case hipSuccess: return "hipSuccess";
        case hipErrorNotSupported: return "hipErrorNotSupported";
        case hipErrorUnsupportedLimit: return "hipErrorUnsupportedLimit";
        case hipErrorInvalidValue: return "hipErrorInvalidValue";
        default: return "other";
    }
}

// ---------------------------------------------------------------------------
// Kernels. Two, differing only in how they access memory, so a difference in
// result is attributable to the access path and nothing else.
// ---------------------------------------------------------------------------
__global__ void caps_kernel_plain(const float *in, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] * 3.0f + 1.0f;
}

__global__ void caps_kernel_nt(const float *in, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = LOCUS_NT_LOAD(&in[i]);
        LOCUS_NT_STORE(v * 3.0f + 1.0f, &out[i]);
    }
}

// Verifies EVERY element. A partial launch is exactly what a spot check
// misses, and a partial launch is the realistic fault here.
static bool run_and_verify(const char *label, void (*)(const float *, float *, int),
                           bool use_nt, int n) {
    std::vector<float> in(n), host_out(n, -1.0f);
    for (int i = 0; i < n; ++i) in[i] = (float)i;

    float *d_in = nullptr, *d_out = nullptr;
    if (hipMalloc(&d_in, n * sizeof(float)) != hipSuccess) return false;
    if (hipMalloc(&d_out, n * sizeof(float)) != hipSuccess) return false;
    if (hipMemcpy(d_in, in.data(), n * sizeof(float), hipMemcpyHostToDevice) !=
        hipSuccess)
        return false;
    if (hipMemcpy(d_out, host_out.data(), n * sizeof(float),
                  hipMemcpyHostToDevice) != hipSuccess)
        return false;

    // Clear any STICKY error left by an earlier deliberate probe before
    // checking this launch. Section 5 calls hipDeviceGetLimit on a limit the
    // platform refuses, and a failed call sets the last-error state -- so
    // hipGetLastError() below would report THAT refusal as if this launch had
    // failed, and then clear it so the NEXT kernel looks fine. Observed
    // exactly that: plain FAILed, non-temporal passed, same arithmetic.
    (void)hipGetLastError();
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    if (use_nt) {
        hipLaunchKernelGGL(caps_kernel_nt, dim3(blocks), dim3(threads), 0, 0,
                           d_in, d_out, n);
    } else {
        hipLaunchKernelGGL(caps_kernel_plain, dim3(blocks), dim3(threads), 0, 0,
                           d_in, d_out, n);
    }
    if (hipGetLastError() != hipSuccess) return false;
    if (hipDeviceSynchronize() != hipSuccess) return false;
    if (hipMemcpy(host_out.data(), d_out, n * sizeof(float),
                  hipMemcpyDeviceToHost) != hipSuccess)
        return false;

    int bad = 0, first_bad = -1;
    for (int i = 0; i < n; ++i) {
        float want = in[i] * 3.0f + 1.0f;
        if (host_out[i] != want) {
            if (first_bad < 0) first_bad = i;
            ++bad;
        }
    }
    (void)hipFree(d_in);
    (void)hipFree(d_out);
    if (bad) {
        out("         %s: %d/%d wrong, first at index %d\n", label, bad, n,
            first_bad);
        return false;
    }
    return true;
}

// Single exit path. Every return from main goes through here, including the
// early failures: a run that died on the first check still WROTE a log, and
// a probe that does not enumerate what it wrote leaves the reader guessing
// whether there is evidence on disk at all.
static int finish(const char *log_path) {
    out("\n[artifacts written]\n");
    out("  %s   (this log; final size printed to stdout after close)\n",
        log_path);
    out("  (no other files written by this probe)\n");
    out("\nRESULT: %s (%d ok, %d failed)\n", g_fail ? "FAIL" : "PASS", g_pass,
        g_fail);

    // Size is stat'd AFTER close. Stat'ing an open, buffered stream reports
    // only what has been flushed -- it read 0 bytes the first time this ran,
    // which is a false claim about the probe's own output.
    if (g_log) {
        std::fclose(g_log);
        g_log = nullptr;
    }
    struct stat st;
    if (stat(log_path, &st) == 0) {
        std::printf("[artifact] %s  %lld bytes\n", log_path,
                    (long long)st.st_size);
    } else {
        std::printf("[artifact] %s  stat FAILED after close\n", log_path);
    }
    return g_fail ? 1 : 0;
}

int main(int argc, char **argv) {
    const char *log_path = (argc > 1) ? argv[1] : "gpu-caps.log";
    g_log = std::fopen(log_path, "w");
    if (!g_log) {
        std::fprintf(stderr, "caps: cannot open log %s\n", log_path);
        return 2;
    }

    out("== locus gpu caps ==\n");
    out("built with: LOCUS_HAS_NT=%d\n", (int)(LOCUS_HAS_NT));

    // -- 1. device present -------------------------------------------------
    int devices = 0;
    hipError_t e = hipGetDeviceCount(&devices);
    out("\n[1] device enumeration\n");
    check(e == hipSuccess && devices > 0, "a HIP device is visible",
          devices > 0 ? "count>0" : "no device -- nothing below is meaningful");
    if (!(e == hipSuccess && devices > 0)) {
        return finish(log_path);
    }

    hipDeviceProp_t p;
    std::memset(&p, 0, sizeof p);
    if (hipGetDeviceProperties(&p, 0) != hipSuccess) {
        check(false, "hipGetDeviceProperties succeeded",
              "without it no field below can be read");
        return finish(log_path);
    }
    out("  device        : %s\n", p.name);
    out("  gcnArchName   : %s\n", p.gcnArchName);

    // -- 2. compute-unit accounting ---------------------------------------
    // rocminfo and HIP disagree by a factor of two on RDNA (two CUs per
    // work-group processor). Read BOTH attributes so the factor is measured
    // here rather than cross-referenced against a different tool.
    out("\n[2] compute units\n");
    int mp = -1, phys = -1;
    hipError_t e_mp =
        hipDeviceGetAttribute(&mp, hipDeviceAttributeMultiprocessorCount, 0);
    hipError_t e_ph = hipDeviceGetAttribute(
        &phys, hipDeviceAttributePhysicalMultiProcessorCount, 0);
    out("  multiProcessorCount (prop)          : %d\n", p.multiProcessorCount);
    out("  hipDeviceAttributeMultiprocessorCount : %d (%s)\n", mp,
        err_name(e_mp));
    out("  ...PhysicalMultiProcessorCount        : %d (%s)\n", phys,
        err_name(e_ph));
    check(p.multiProcessorCount > 0, "multiProcessorCount > 0",
          "used as the occupancy denominator");
    if (e_ph == hipSuccess && phys > 0 && mp > 0) {
        out("  physical/logical ratio              : %.2f\n",
            (double)phys / (double)mp);
    }

    // -- 3. the hierarchy, innermost outward ------------------------------
    out("\n[3] memory hierarchy as HIP reports it\n");
    out("  regsPerBlock                        : %d (32-bit regs)\n",
        p.regsPerBlock);
    out("  regsPerMultiprocessor               : %lld\n",
        locus_field_regsPerMultiprocessor(p));
    out("  warpSize                            : %d\n", p.warpSize);
    out("  sharedMemPerBlock (LDS)             : %zu bytes\n",
        (size_t)p.sharedMemPerBlock);
    out("  reservedSharedMemPerBlock           : %lld\n",
        locus_field_reservedSharedMemPerBlock(p));
    out("  maxSharedMemoryPerMultiProcessor    : %lld\n",
        locus_field_maxSharedMemoryPerMultiProcessor(p));
    out("  globalL1CacheSupported              : %lld\n",
        locus_field_globalL1CacheSupported(p));
    out("  l2CacheSize                         : %d bytes\n", p.l2CacheSize);
    out("  totalGlobalMem                      : %.2f GiB\n",
        (double)p.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    out("  memoryBusWidth / memoryClockRate    : %d bits / %d kHz\n",
        p.memoryBusWidth, p.memoryClockRate);
    if (p.memoryBusWidth > 0 && p.memoryClockRate > 0) {
        // Two candidates, deliberately BOTH printed. Whether HIP reports
        // memoryClockRate as a base clock or as the effective per-pin data
        // rate differs by device, and picking one here would state a
        // measurement and its explanation as if they were one fact. Compare
        // against the card's published figure to settle which applies; do
        // not carry the wrong one forward as a bandwidth budget.
        double bytes_per_transfer = (double)p.memoryBusWidth / 8.0;
        double hz = (double)p.memoryClockRate * 1000.0;
        out("  DRAM bw if clock is EFFECTIVE rate  : %.1f GB/s\n",
            bytes_per_transfer * hz / 1e9);
        out("  DRAM bw if clock is base, DDR x2    : %.1f GB/s\n",
            bytes_per_transfer * hz * 2.0 / 1e9);
    }
    out("  NOTE: HIP exposes no L0, no L1 SIZE and no L3/MALL field at all.\n");
    out("        L1 and Infinity Cache come from rocminfo only.\n");

    check(p.warpSize == 32 || p.warpSize == 64, "warpSize is 32 or 64",
          "any other value means the probe misread the struct");
    check(p.sharedMemPerBlock >= 16384, "LDS per block >= 16 KiB",
          "below this the tier design has no on-chip scratch to use");
    check(p.totalGlobalMem > 0, "totalGlobalMem > 0",
          "the VRAM budget every tier decision is sized against");

    // -- 4. per-kernel resource use ---------------------------------------
    // The device totals above are a ceiling; what a kernel actually spends
    // is the number that decides occupancy and whether state spills to
    // scratch. localSizeBytes > 0 IS the spill.
    out("\n[4] per-kernel resource use (hipFuncGetAttributes)\n");
    hipFuncAttributes fa;
    std::memset(&fa, 0, sizeof fa);
    hipError_t e_fa =
        hipFuncGetAttributes(&fa, reinterpret_cast<const void *>(caps_kernel_plain));
    if (e_fa == hipSuccess) {
        out("  numRegs                             : %d\n", fa.numRegs);
        out("  localSizeBytes (scratch spill)      : %zu\n",
            (size_t)fa.localSizeBytes);
        out("  sharedSizeBytes (static LDS)        : %zu\n",
            (size_t)fa.sharedSizeBytes);
        out("  maxThreadsPerBlock                  : %d\n",
            fa.maxThreadsPerBlock);
        check(fa.numRegs > 0, "numRegs reported for a real kernel",
              "zero means the query did not reach the loaded code object");
    } else {
        out("  hipFuncGetAttributes: %s\n", err_name(e_fa));
        check(false, "hipFuncGetAttributes succeeded", err_name(e_fa));
    }

    // -- 5. is any of the hierarchy runtime-configurable? -----------------
    // This section exists to TEST a claim rather than to gather a number.
    // Result is reported, not asserted: either answer is a legitimate
    // property of the platform, and asserting one would bake a platform
    // assumption into the probe.
    out("\n[5] cache-control surface (claims under test)\n");
    hipError_t e_cc = hipDeviceSetCacheConfig(hipFuncCachePreferShared);
    out("  hipDeviceSetCacheConfig              : %s\n", err_name(e_cc));

    struct { const char *n; enum hipLimit_t l; } limits[] = {
        {"hipLimitStackSize", hipLimitStackSize},
        {"hipLimitPrintfFifoSize", hipLimitPrintfFifoSize},
        {"hipLimitMallocHeapSize", hipLimitMallocHeapSize},
    };
    for (auto &lim : limits) {
        size_t v = 0;
        hipError_t g = hipDeviceGetLimit(&v, lim.l);
        out("  get %-24s        : %-24s value=%zu\n", lim.n, err_name(g), v);
    }
    out("  hipLimit_t has NO persisting-cache member, so hipAccessPolicyWindow\n");
    out("  has no limit to arm it. Reported below for completeness:\n");
    out("  persistingL2CacheMaxSize            : %lld\n",
        locus_field_persistingL2CacheMaxSize(p));
    out("  accessPolicyMaxWindowSize           : %lld\n",
        locus_field_accessPolicyMaxWindowSize(p));

    // -- 6. non-temporal access -------------------------------------------
    // The only positive control that exists is telling the cache what NOT to
    // keep. Correctness first: a hint that changes results is a bug, not a
    // hint.
    out("\n[6] non-temporal access\n");
    const int n = 1 << 20;
    bool plain_ok = run_and_verify("plain", nullptr, false, n);
    check(plain_ok, "plain kernel: every element verified",
          "1Mi elements, distinct expected value each");
    bool nt_ok = run_and_verify("nontemporal", nullptr, true, n);
    check(nt_ok, LOCUS_HAS_NT ? "nontemporal builtins: results identical"
                              : "nontemporal FALLBACK: results identical",
          LOCUS_HAS_NT ? "builtins compiled in"
                       : "builtins unavailable, plain path used");

    // -- 7. persistent-kernel preconditions -------------------------------
    // Reported only. Actually running a long kernel is a separate,
    // deliberately gated artifact -- on a card also driving a display a
    // watchdog reset is a real consequence.
    out("\n[7] persistent-kernel preconditions (reported, not exercised)\n");
    out("  cooperativeLaunch                   : %lld\n",
        locus_field_cooperativeLaunch(p));
    out("  kernelExecTimeoutEnabled            : %lld\n",
        locus_field_kernelExecTimeoutEnabled(p));
    out("  concurrentKernels                   : %d\n", p.concurrentKernels);
    out("  maxThreadsPerMultiProcessor         : %d\n",
        p.maxThreadsPerMultiProcessor);
    out("  see persist.cpp, gated behind LOCUS_GPU_STRESS=1\n");

    // -- artifacts ---------------------------------------------------------
    // A tool's returned text is a summary; the files it wrote are the
    // evidence, and a summary never says what it left out. So enumerate
    // every path written, with sizes, including this log.
    return finish(log_path);
}
