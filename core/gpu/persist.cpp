// Persistent-kernel viability probe.
//
// THE QUESTION: LDS is the only on-chip memory under software control, but
// it does not survive a kernel launch. So an activity-maintained ACTIVE tier
// that genuinely lives "where the compute is" requires a kernel that stays
// resident across ticks. This probe establishes whether that is available
// here and at what occupancy cost -- it does not build one.
//
// *** THIS PROBE CAN COST YOU THE RUNNING GRAPHICAL SESSION. ***
// A long-running kernel on a GPU that is also driving a display can trip the
// driver's timeout-detection-and-recovery path. What that resets is the GPU
// context: the display session and any resident compute state (a loaded
// model, another process's device allocations) go with it. Nothing on disk
// is touched -- this probe writes only its own log, and the repository and
// every file are unaffected -- but unsaved work in the graphical session is
// not recoverable from anywhere. Do not run it with a model loaded, and do
// not run it with unsaved work open.
//
// Because of that it is gated: it SKIPS with exit 0 unless
// LOCUS_GPU_STRESS=1 is set explicitly. The tick count starts deliberately
// low and is raised by hand via LOCUS_GPU_STRESS_TICKS -- the duration is
// reported so the ladder is climbed on evidence rather than in one jump.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/stat.h>
#include <vector>

#ifndef LOCUS_PERSIST_BLOCK
#  define LOCUS_PERSIST_BLOCK 256
#endif

static std::FILE *g_log = nullptr;
static int g_pass = 0, g_fail = 0;
// A gated run performs no checks. Reporting PASS for that would claim a
// verdict it never reached, so the skip is reported as SKIP.
static bool g_skipped = false;

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

static void check(bool ok, const char *what, const char *detail) {
    if (ok) { ++g_pass; out("  [ok]   %-38s %s\n", what, detail); }
    else    { ++g_fail; out("  [FAIL] %-38s %s\n", what, detail); }
}

// Each thread owns one LDS slot and advances only its OWN slot across ticks.
// __syncthreads() is still called every tick, so the barrier is exercised on
// real hardware -- but the arithmetic deliberately does NOT depend on
// cross-thread coupling, which is what lets a host-side mock verify the
// numbers. A barrier-DEPENDENT variant (each thread reading its neighbour's
// previous tick) is a real-device-only test and is not this file's job:
// asserting it here would mean asserting something the mock cannot model.
__global__ void persist_kernel(float *out_buf, int ticks, int n) {
    __shared__ float lds[LOCUS_PERSIST_BLOCK];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < LOCUS_PERSIST_BLOCK) lds[tid] = 0.0f;
    __syncthreads();

    for (int t = 0; t < ticks; ++t) {
        if (tid < LOCUS_PERSIST_BLOCK) lds[tid] += 1.0f;
        __syncthreads();
    }

    if (gid < n && tid < LOCUS_PERSIST_BLOCK) out_buf[gid] = lds[tid];
}

static int finish(const char *log_path) {
    out("\n[artifacts written]\n");
    out("  %s   (this log; final size printed to stdout after close)\n",
        log_path);
    out("  (no other files written by this probe)\n");
    out("\nRESULT: %s (%d ok, %d failed)\n",
        g_skipped ? "SKIP" : (g_fail ? "FAIL" : "PASS"), g_pass, g_fail);
    if (g_log) { std::fclose(g_log); g_log = nullptr; }
    struct stat st;
    if (stat(log_path, &st) == 0)
        std::printf("[artifact] %s  %lld bytes\n", log_path,
                    (long long)st.st_size);
    else
        std::printf("[artifact] %s  stat FAILED after close\n", log_path);
    return g_fail ? 1 : 0;
}

int main(int argc, char **argv) {
    const char *log_path = (argc > 1) ? argv[1] : "gpu-persist.log";
    g_log = std::fopen(log_path, "w");
    if (!g_log) {
        std::fprintf(stderr, "persist: cannot open log %s\n", log_path);
        return 2;
    }
    out("== locus gpu persist ==\n");

    const char *gate = std::getenv("LOCUS_GPU_STRESS");
    if (!gate || std::strcmp(gate, "1") != 0) {
        out("\n[SKIP] LOCUS_GPU_STRESS is not 1.\n");
        out("  This probe runs a deliberately long kernel. On a GPU that is\n");
        out("  also driving a display that can trigger a driver reset, which\n");
        out("  loses the graphical session and any resident model. No file is\n");
        out("  affected either way. Set LOCUS_GPU_STRESS=1 to run it, with\n");
        out("  no model loaded and no unsaved work open.\n");
        g_skipped = true;
        return finish(log_path);  // SKIP is not a failure
    }

    int ticks = 1000;  // deliberately low; raise by hand on evidence
    if (const char *tv = std::getenv("LOCUS_GPU_STRESS_TICKS")) {
        int parsed = std::atoi(tv);
        if (parsed > 0) ticks = parsed;
    }

    int devices = 0;
    if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) {
        check(false, "a HIP device is visible", "no device");
        return finish(log_path);
    }
    hipDeviceProp_t p;
    std::memset(&p, 0, sizeof p);
    if (hipGetDeviceProperties(&p, 0) != hipSuccess) {
        check(false, "hipGetDeviceProperties succeeded", "");
        return finish(log_path);
    }
    out("  device : %s (%s)\n", p.name, p.gcnArchName);
    out("  ticks  : %d, block %d\n", ticks, LOCUS_PERSIST_BLOCK);

    // -- preconditions, read before anything long runs --------------------
    out("\n[1] preconditions\n");
    int coop = -1, wallrate = -1;
    hipError_t e_coop =
        hipDeviceGetAttribute(&coop, hipDeviceAttributeCooperativeLaunch, 0);
    hipError_t e_wall =
        hipDeviceGetAttribute(&wallrate, hipDeviceAttributeWallClockRate, 0);
    out("  cooperative launch supported        : %d (%s)\n", coop,
        e_coop == hipSuccess ? "ok" : "query failed");
    out("  wall clock rate                     : %d kHz (%s)\n", wallrate,
        e_wall == hipSuccess ? "ok" : "query failed");

    // Occupancy is the real cost of a resident kernel: whatever it holds is
    // unavailable to anything else for as long as it lives.
    int blocks_per_cu = 0;
    hipError_t e_occ = hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_cu, reinterpret_cast<const void *>(persist_kernel),
        LOCUS_PERSIST_BLOCK, 0);
    out("  max active blocks per multiprocessor: %d (%s)\n", blocks_per_cu,
        e_occ == hipSuccess ? "ok" : "query failed");
    check(e_occ == hipSuccess && blocks_per_cu > 0,
          "occupancy query returned a grid bound",
          "without it a cooperative grid size cannot be chosen safely");
    if (!(e_occ == hipSuccess && blocks_per_cu > 0)) return finish(log_path);

    int grid = blocks_per_cu * (p.multiProcessorCount > 0 ? p.multiProcessorCount : 1);
    int n = grid * LOCUS_PERSIST_BLOCK;
    out("  resident grid this device supports  : %d blocks, %d threads\n",
        grid, n);

    // -- 2. ordinary launch, LDS state across ticks ------------------------
    out("\n[2] LDS state across ticks (ordinary launch)\n");
    std::vector<float> host(n, -1.0f);
    float *dev = nullptr;
    if (hipMalloc(&dev, (size_t)n * sizeof(float)) != hipSuccess) {
        check(false, "device allocation", "hipMalloc failed");
        return finish(log_path);
    }
    hipMemcpy(dev, host.data(), (size_t)n * sizeof(float),
              hipMemcpyHostToDevice);

    // Timing here is a SAFETY budget, not a performance measurement: it is
    // how the tick ladder gets climbed deliberately. It says nothing about
    // throughput and must not be quoted as if it did.
    auto t0 = std::chrono::steady_clock::now();
    hipLaunchKernelGGL(persist_kernel, dim3(grid), dim3(LOCUS_PERSIST_BLOCK), 0,
                       0, dev, ticks, n);
    hipError_t e_launch = hipGetLastError();
    hipError_t e_sync = hipDeviceSynchronize();
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    out("  wall duration                       : %.1f ms for %d ticks\n", ms,
        ticks);
    check(e_launch == hipSuccess && e_sync == hipSuccess,
          "kernel completed without a device error",
          "a reset or timeout surfaces here");

    hipMemcpy(host.data(), dev, (size_t)n * sizeof(float),
              hipMemcpyDeviceToHost);
    int bad = 0, first_bad = -1;
    for (int i = 0; i < n; ++i) {
        if (host[i] != (float)ticks) {
            if (first_bad < 0) first_bad = i;
            ++bad;
        }
    }
    if (bad)
        out("         %d/%d slots wrong, first at %d (got %.1f, want %d)\n",
            bad, n, first_bad, host[first_bad], ticks);
    check(bad == 0, "every LDS slot survived every tick",
          "each slot must equal the tick count exactly");

    // -- 3. cooperative launch --------------------------------------------
    // The mechanism a real persistent kernel needs: a grid guaranteed
    // co-resident, so a grid-wide barrier is legal.
    out("\n[3] cooperative launch\n");
    if (coop != 1) {
        out("  [SKIP] device reports cooperative launch unsupported\n");
    } else {
        std::fill(host.begin(), host.end(), -1.0f);
        hipMemcpy(dev, host.data(), (size_t)n * sizeof(float),
                  hipMemcpyHostToDevice);
        void *args[] = {&dev, &ticks, &n};
        hipError_t e_cl = hipLaunchCooperativeKernel(
            reinterpret_cast<const void *>(persist_kernel), dim3(grid),
            dim3(LOCUS_PERSIST_BLOCK), args, 0, 0);
        hipError_t e_cs = hipDeviceSynchronize();
        out("  hipLaunchCooperativeKernel          : %s\n",
            e_cl == hipSuccess ? "accepted" : "rejected");
        check(e_cl == hipSuccess && e_cs == hipSuccess,
              "cooperative launch accepted at full grid",
              "if rejected, the resident grid must be sized down");
        if (e_cl == hipSuccess && e_cs == hipSuccess) {
            hipMemcpy(host.data(), dev, (size_t)n * sizeof(float),
                      hipMemcpyDeviceToHost);
            int cbad = 0;
            for (int i = 0; i < n; ++i)
                if (host[i] != (float)ticks) ++cbad;
            check(cbad == 0, "cooperative result matches ordinary launch",
                  "same arithmetic, so any difference is the launch path");
        }
    }

    hipFree(dev);
    return finish(log_path);
}
