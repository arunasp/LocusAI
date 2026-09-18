// Excursion structure of a heavy-tailed kernel, measured on the device.
//
// WHY: every claim about how deep the trace-chain recursion must go has so
// far come from pure-Python toys (n <= 128) or from reasoning, and both
// have a poor record here. This measures two things at a size where the
// dropped block no longer fits a cache:
//
//   1. rho(P_BB) by power iteration. If it reaches 1 the complement is
//      CLOSED and no trace chain exists for that partition, at any depth.
//      That was measured on 48-state toys; this checks it at n=4096.
//   2. The EXACT excursion-return distribution: starting from the average
//      leak out of the kept set, the mass returning after exactly s steps
//      in the complement. That distribution's tail is what sets the
//      truncation depth, and here it is computed from the kernel rather
//      than sampled from a stream.
//
// Two partitions are compared because partition CHOICE was measured to
// matter more than kernel density: top-frequency (hubs kept in A) versus
// random (hubs mostly in the complement). A prediction on record, from the
// character-corpus run, is that technical-text-like heavy tails give a
// BIMODAL return distribution; if so, one depth per level is not enough
// and the estimator needs regime detection before it means anything.
//
// Deliberately no throughput claims. The kernels below are written for
// clarity, and the transpose is passed explicitly rather than strided, so
// timings here bound the measurement cost, not the achievable cost.

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define HIP_OK(x)                                                        \
    do {                                                                 \
        hipError_t e_ = (x);                                             \
        if (e_ != hipSuccess) {                                          \
            std::printf("HIP fail %s:%d -> %d\n", __FILE__, __LINE__,    \
                        (int)e_);                                        \
            std::exit(1);                                                \
        }                                                                \
    } while (0)

// y = A x, A row-major n*n, one thread per row so each thread walks a
// contiguous row. The caller passes an explicit transpose when it needs
// the other orientation rather than striding here.
__global__ void matvec(const float *a, const float *x, float *y, int n) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n) return;
    const float *row = a + (size_t)r * n;
    float s = 0.0f;
    for (int c = 0; c < n; ++c) s += row[c] * x[c];
    y[r] = s;
}

__global__ void scale_by(float *v, int n, const float *m) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && *m > 0.0f) v[i] /= *m;
}

// Single-block reductions: n is a few thousand, so one block suffices and
// it avoids a second launch per iteration.
__global__ void reduce_absmax(const float *v, int n, float *out) {
    __shared__ float s[256];
    float m = 0.0f;
    for (int i = threadIdx.x; i < n; i += 256) m = fmaxf(m, fabsf(v[i]));
    s[threadIdx.x] = m;
    __syncthreads();
    for (int o = 128; o > 0; o >>= 1) {
        if (threadIdx.x < o)
            s[threadIdx.x] = fmaxf(s[threadIdx.x], s[threadIdx.x + o]);
        __syncthreads();
    }
    if (threadIdx.x == 0) *out = s[0];
}

__global__ void reduce_dot(const float *a, const float *b, int n,
                           float *out) {
    __shared__ float s[256];
    float t = 0.0f;
    for (int i = threadIdx.x; i < n; i += 256) t += a[i] * b[i];
    s[threadIdx.x] = t;
    __syncthreads();
    for (int o = 128; o > 0; o >>= 1) {
        if (threadIdx.x < o) s[threadIdx.x] += s[threadIdx.x + o];
        __syncthreads();
    }
    if (threadIdx.x == 0) *out = s[0];
}

int main(int argc, char **argv) {
    const int n = (argc > 1) ? std::atoi(argv[1]) : 4096;
    const int na = (argc > 2) ? std::atoi(argv[2]) : 1024;

    // Heavy-tailed kernel standing in for technical text: small skewed
    // out-degree, targets drawn with a cubic bias toward low indices so
    // low indices behave as high-frequency hubs. Index order therefore IS
    // frequency rank, which is what makes the two partitions comparable.
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> U(0.0f, 1.0f);
    std::vector<float> P((size_t)n * n, 0.0f);
    for (int i = 0; i < n; ++i) {
        float *row = &P[(size_t)i * n];
        row[i] = 0.2f + 0.3f * U(rng);
        int deg = 2 + (int)(18.0f * std::pow(U(rng), 2.5f));
        for (int d = 0; d < deg; ++d) {
            float u = U(rng);
            int t = (int)(n * u * u * u);
            if (t >= n) t = n - 1;
            row[t] += U(rng);
        }
        float s = 0.0f;
        for (int c = 0; c < n; ++c) s += row[c];
        for (int c = 0; c < n; ++c) row[c] /= s;
    }

    for (int mode = 0; mode < 2; ++mode) {
        std::vector<char> inA(n, 0);
        if (mode == 0) {
            for (int i = 0; i < na; ++i) inA[i] = 1;     // hubs kept
        } else {
            std::mt19937 r2(777);                        // hubs dropped
            int placed = 0;
            while (placed < na) {
                int i = (int)(r2() % (unsigned)n);
                if (!inA[i]) { inA[i] = 1; ++placed; }
            }
        }
        std::vector<int> A, B;
        for (int i = 0; i < n; ++i) (inA[i] ? A : B).push_back(i);
        const int nb = (int)B.size();

        // P_BB and its transpose; the leak out of A averaged over A's
        // rows; and the return mass from each complement state into A.
        std::vector<float> Pbb((size_t)nb * nb), PbbT((size_t)nb * nb);
        std::vector<float> leak(nb, 0.0f), ret(nb, 0.0f);
        for (int r = 0; r < nb; ++r) {
            for (int c = 0; c < nb; ++c) {
                float v = P[(size_t)B[r] * n + B[c]];
                Pbb[(size_t)r * nb + c] = v;
                PbbT[(size_t)c * nb + r] = v;
            }
            float s = 0.0f;
            for (size_t a = 0; a < A.size(); ++a)
                s += P[(size_t)B[r] * n + A[a]];
            ret[r] = s;
        }
        for (size_t a = 0; a < A.size(); ++a)
            for (int c = 0; c < nb; ++c)
                leak[c] += P[(size_t)A[a] * n + B[c]] / (float)A.size();

        float *d_bb, *d_bbT, *d_u, *d_t, *d_ret, *d_s;
        size_t bytes = (size_t)nb * nb * sizeof(float);
        HIP_OK(hipMalloc(&d_bb, bytes));
        HIP_OK(hipMalloc(&d_bbT, bytes));
        HIP_OK(hipMalloc(&d_u, nb * sizeof(float)));
        HIP_OK(hipMalloc(&d_t, nb * sizeof(float)));
        HIP_OK(hipMalloc(&d_ret, nb * sizeof(float)));
        HIP_OK(hipMalloc(&d_s, sizeof(float)));
        HIP_OK(hipMemcpy(d_bb, Pbb.data(), bytes, hipMemcpyHostToDevice));
        HIP_OK(hipMemcpy(d_bbT, PbbT.data(), bytes,
                         hipMemcpyHostToDevice));
        HIP_OK(hipMemcpy(d_ret, ret.data(), nb * sizeof(float),
                         hipMemcpyHostToDevice));

        int blocks = (nb + 255) / 256;
        std::printf("\n=== n=%d |A|=%d |B|=%d partition=%s "
                    "P_BB=%.1f MiB ===\n", n, (int)A.size(), nb,
                    mode == 0 ? "top-frequency" : "random",
                    bytes / 1048576.0);

        // rho(P_BB). Reaching 1 means the complement is closed and the
        // trace chain does not exist for this partition.
        std::vector<float> ones(nb, 1.0f);
        HIP_OK(hipMemcpy(d_u, ones.data(), nb * sizeof(float),
                         hipMemcpyHostToDevice));
        hipEvent_t e0, e1;
        HIP_OK(hipEventCreate(&e0));
        HIP_OK(hipEventCreate(&e1));
        HIP_OK(hipEventRecord(e0));
        for (int it = 0; it < 300; ++it) {
            hipLaunchKernelGGL(matvec, dim3(blocks), dim3(256), 0, 0,
                               d_bb, d_u, d_t, nb);
            hipLaunchKernelGGL(reduce_absmax, dim3(1), dim3(256), 0, 0,
                               d_t, nb, d_s);
            hipLaunchKernelGGL(scale_by, dim3(blocks), dim3(256), 0, 0,
                               d_t, nb, d_s);
            float *tmp = d_u; d_u = d_t; d_t = tmp;
        }
        hipLaunchKernelGGL(matvec, dim3(blocks), dim3(256), 0, 0,
                           d_bb, d_u, d_t, nb);
        float num = 0.0f, den = 0.0f;
        hipLaunchKernelGGL(reduce_dot, dim3(1), dim3(256), 0, 0, d_t, d_u,
                           nb, d_s);
        HIP_OK(hipMemcpy(&num, d_s, sizeof(float), hipMemcpyDeviceToHost));
        hipLaunchKernelGGL(reduce_dot, dim3(1), dim3(256), 0, 0, d_u, d_u,
                           nb, d_s);
        HIP_OK(hipMemcpy(&den, d_s, sizeof(float), hipMemcpyDeviceToHost));
        HIP_OK(hipEventRecord(e1));
        HIP_OK(hipEventSynchronize(e1));
        float ms_rho = 0.0f;
        HIP_OK(hipEventElapsedTime(&ms_rho, e0, e1));
        float rho = (den > 0.0f) ? num / den : 0.0f;
        std::printf("rho(P_BB) = %.5f   (300 power iterations, %.1f ms)\n",
                    rho, ms_rho);
        if (rho > 0.999f)
            std::printf("  WARNING: complement effectively closed -- no "
                        "trace chain exists for this partition\n");

        // Exact excursion-return distribution. u starts as the average
        // leak profile; each step is u <- u P_BB, done as PbbT * u so
        // reads stay contiguous. r[s] is mass returning after s+1 steps.
        HIP_OK(hipMemcpy(d_u, leak.data(), nb * sizeof(float),
                         hipMemcpyHostToDevice));
        std::vector<double> r;
        HIP_OK(hipEventRecord(e0));
        for (int s = 0; s < 16; ++s) {
            hipLaunchKernelGGL(reduce_dot, dim3(1), dim3(256), 0, 0, d_u,
                               d_ret, nb, d_s);
            float got = 0.0f;
            HIP_OK(hipMemcpy(&got, d_s, sizeof(float),
                             hipMemcpyDeviceToHost));
            r.push_back(got);
            hipLaunchKernelGGL(matvec, dim3(blocks), dim3(256), 0, 0,
                               d_bbT, d_u, d_t, nb);
            float *tmp = d_u; d_u = d_t; d_t = tmp;
        }
        HIP_OK(hipEventRecord(e1));
        HIP_OK(hipEventSynchronize(e1));
        float ms_exc = 0.0f;
        HIP_OK(hipEventElapsedTime(&ms_exc, e0, e1));

        double tot = 0.0;
        for (double v : r) tot += v;
        std::printf("return distribution (%.1f ms), total returned "
                    "%.4f of leak\n", ms_exc, tot);
        std::printf("  steps  frac     cumulative\n");
        double cum = 0.0, mean = 0.0;
        int kneed = -1;
        for (size_t s = 0; s < r.size(); ++s) {
            double frac = (tot > 0.0) ? r[s] / tot : 0.0;
            cum += frac;
            mean += frac * (double)(s + 1);
            if (s < 8 || frac > 0.01)
                std::printf("  %-6zu %-8.4f %.4f\n", s + 1, frac, cum);
            if (kneed < 0 && cum >= 0.99) kneed = (int)s;
        }
        std::printf("mean excursion %.3f steps; depth for 99%% of the "
                    "mass: %d\n", mean, kneed);

        HIP_OK(hipFree(d_bb));
        HIP_OK(hipFree(d_bbT));
        HIP_OK(hipFree(d_u));
        HIP_OK(hipFree(d_t));
        HIP_OK(hipFree(d_ret));
        HIP_OK(hipFree(d_s));
        HIP_OK(hipEventDestroy(e0));
        HIP_OK(hipEventDestroy(e1));
    }
    std::printf("\nRESULT: measured\n");
    return 0;
}
