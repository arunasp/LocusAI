// Per-input stream learners on the GPU (or CPU without hipcc).
//
// The per-input learners in locus.learn (BranchLearner, the CLS stores)
// update each unit's row from that unit's own error only, so every unit's
// events can be replayed independently in their original order. This
// program does exactly that: one block per unit, one thread per byte, the
// weight held in a register across the unit's events. It then reads out
// every scored position: one block per position, one thread per byte.
//
// Arithmetic is fp64 and in Python's order, w + ((rate * err) * gain) *
// local, skipping dw == 0, so the weights are meant to equal the Python
// learners'. Checked through scores: built without hipcc, held-out bits
// per byte equal the Python learners' to 12 decimals on a frozen corpus.
// The positive-drive sum is a tree reduction, so scores may differ from
// Python in the last bits.
//
// Usage: learn_stream IN OUT
//   IN : int32 U; double rate; int32 S (stores, 1 or 2); per store:
//        int64 rowptr[U+1], int64 E, uint8 nxt[E], double local[E];
//        int64 P, int64 pptr[P+1], int32 punit[pptr[P]],
//        uint8 phippo[pptr[P]], uint8 pnext[P]
//   OUT: double dnext[P], double dpos[P]
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#ifdef __HIPCC__
#include <hip/hip_runtime.h>
#define CHECK(x)                                                          \
    do {                                                                  \
        hipError_t e_ = (x);                                              \
        if (e_ != hipSuccess) {                                           \
            std::fprintf(stderr, "FAIL: %s: %s\n", #x,                    \
                         hipGetErrorString(e_));                          \
            std::exit(1);                                                 \
        }                                                                 \
    } while (0)
#define HD __host__ __device__
#else
#define HD
#endif

static const int B = 256;

// One event of one unit, for byte b. Mirrors Plasticity.observe_error +
// consolidate(gain=1, consume=True, local=loc): skip when dw == 0, clamp
// at +-w_max (5.0).
HD inline double step(double w, int b, int nxt, double rate, double loc)
{
    double err = (b == nxt ? 1.0 : 0.0) - w;
    if (err == 0.0)
        return w;
    double dw = rate * err;
    dw = dw * 1.0;
    dw = dw * loc;
    if (dw == 0.0)
        return w;
    w = w + dw;
    if (w > 5.0)
        w = 5.0;
    else if (w < -5.0)
        w = -5.0;
    return w;
}

#ifdef __HIPCC__
__global__ void learn_kernel(const int64_t *rowptr, const uint8_t *nxt,
                             const double *loc, double rate, double *W)
{
    int u = blockIdx.x, b = threadIdx.x;
    double w = 0.0;
    for (int64_t e = rowptr[u]; e < rowptr[u + 1]; ++e)
        w = step(w, b, nxt[e], rate, loc[e]);
    W[(int64_t)u * B + b] = w;
}

__global__ void score_kernel(const int64_t *pptr, const int32_t *punit,
                             const uint8_t *phippo, const uint8_t *pnext,
                             const double *Wc, const double *Wh,
                             double *dnext, double *dpos)
{
    __shared__ double sh[B];
    int p = blockIdx.x, b = threadIdx.x;
    double dc = 0.0, dh = 0.0;
    for (int64_t j = pptr[p]; j < pptr[p + 1]; ++j) {
        int64_t u = punit[j];
        dc = dc + Wc[u * B + b];
        if (Wh && phippo[j])
            dh = dh + Wh[u * B + b];
    }
    double d = Wh ? dc + dh : dc;
    sh[b] = d > 0.0 ? d : 0.0;
    __syncthreads();
    if (b == pnext[p])
        dnext[p] = sh[b];
    for (int s = B / 2; s > 0; s >>= 1) {
        if (b < s)
            sh[b] += sh[b + s];
        __syncthreads();
    }
    if (b == 0)
        dpos[p] = sh[0];
}
#endif

template <class T>
static void rd(FILE *f, T *p, size_t n)
{
    if (n && std::fread(p, sizeof(T), n, f) != n) {
        std::fprintf(stderr, "FAIL: short read\n");
        std::exit(1);
    }
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        std::fprintf(stderr, "usage: learn_stream IN OUT\n");
        return 2;
    }
    FILE *f = std::fopen(argv[1], "rb");
    if (!f) {
        std::fprintf(stderr, "FAIL: open %s\n", argv[1]);
        return 1;
    }
    int32_t U, S;
    double rate;
    rd(f, &U, 1);
    rd(f, &rate, 1);
    rd(f, &S, 1);
    std::vector<std::vector<int64_t>> rowptr(S);
    std::vector<std::vector<uint8_t>> nxt(S);
    std::vector<std::vector<double>> loc(S);
    for (int s = 0; s < S; ++s) {
        rowptr[s].resize(U + 1);
        rd(f, rowptr[s].data(), U + 1);
        int64_t E;
        rd(f, &E, 1);
        nxt[s].resize(E);
        loc[s].resize(E);
        rd(f, nxt[s].data(), E);
        rd(f, loc[s].data(), E);
    }
    int64_t P;
    rd(f, &P, 1);
    std::vector<int64_t> pptr(P + 1);
    rd(f, pptr.data(), P + 1);
    std::vector<int32_t> punit(pptr[P]);
    std::vector<uint8_t> phippo(pptr[P]), pnext(P);
    rd(f, punit.data(), punit.size());
    rd(f, phippo.data(), phippo.size());
    rd(f, pnext.data(), P);
    std::fclose(f);

    std::vector<double> dnext(P, 0.0), dpos(P, 0.0);
#ifdef __HIPCC__
    std::vector<double *> W(S, nullptr);
    for (int s = 0; s < S; ++s) {
        int64_t *drow;
        uint8_t *dnxt;
        double *dloc;
        CHECK(hipMalloc(&drow, (U + 1) * sizeof(int64_t)));
        CHECK(hipMalloc(&dnxt, nxt[s].size() + 1));
        CHECK(hipMalloc(&dloc, (loc[s].size() + 1) * sizeof(double)));
        CHECK(hipMalloc(&W[s], (size_t)U * B * sizeof(double)));
        CHECK(hipMemcpy(drow, rowptr[s].data(), (U + 1) * sizeof(int64_t),
                        hipMemcpyHostToDevice));
        CHECK(hipMemcpy(dnxt, nxt[s].data(), nxt[s].size(),
                        hipMemcpyHostToDevice));
        CHECK(hipMemcpy(dloc, loc[s].data(), loc[s].size() * sizeof(double),
                        hipMemcpyHostToDevice));
        learn_kernel<<<U, B>>>(drow, dnxt, dloc, rate, W[s]);
        CHECK(hipGetLastError());
        CHECK(hipDeviceSynchronize());
        (void)hipFree(drow);
        (void)hipFree(dnxt);
        (void)hipFree(dloc);
    }
    int64_t *dpptr;
    int32_t *dpunit;
    uint8_t *dph, *dpn;
    double *ddn, *ddp;
    CHECK(hipMalloc(&dpptr, (P + 1) * sizeof(int64_t)));
    CHECK(hipMalloc(&dpunit, (punit.size() + 1) * sizeof(int32_t)));
    CHECK(hipMalloc(&dph, phippo.size() + 1));
    CHECK(hipMalloc(&dpn, P + 1));
    CHECK(hipMalloc(&ddn, P * sizeof(double)));
    CHECK(hipMalloc(&ddp, P * sizeof(double)));
    CHECK(hipMemcpy(dpptr, pptr.data(), (P + 1) * sizeof(int64_t),
                    hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dpunit, punit.data(), punit.size() * sizeof(int32_t),
                    hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dph, phippo.data(), phippo.size(), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dpn, pnext.data(), P, hipMemcpyHostToDevice));
    CHECK(hipMemset(ddn, 0, P * sizeof(double)));
    score_kernel<<<P, B>>>(dpptr, dpunit, dph, dpn, W[0],
                           S > 1 ? W[1] : nullptr, ddn, ddp);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
    CHECK(hipMemcpy(dnext.data(), ddn, P * sizeof(double),
                    hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(dpos.data(), ddp, P * sizeof(double),
                    hipMemcpyDeviceToHost));
#else
    std::vector<std::vector<double>> W(S, std::vector<double>((size_t)U * B));
    for (int s = 0; s < S; ++s)
        for (int u = 0; u < U; ++u)
            for (int b = 0; b < B; ++b) {
                double w = 0.0;
                for (int64_t e = rowptr[s][u]; e < rowptr[s][u + 1]; ++e)
                    w = step(w, b, nxt[s][e], rate, loc[s][e]);
                W[s][(size_t)u * B + b] = w;
            }
    for (int64_t p = 0; p < P; ++p) {
        double total = 0.0;
        for (int b = 0; b < B; ++b) {
            double dc = 0.0, dh = 0.0;
            for (int64_t j = pptr[p]; j < pptr[p + 1]; ++j) {
                size_t u = punit[j];
                dc = dc + W[0][u * B + b];
                if (S > 1 && phippo[j])
                    dh = dh + W[1][u * B + b];
            }
            double d = S > 1 ? dc + dh : dc;
            double v = d > 0.0 ? d : 0.0;
            if (b == pnext[p])
                dnext[p] = v;
            total += v;
        }
        dpos[p] = total;
    }
#endif
    FILE *o = std::fopen(argv[2], "wb");
    if (!o || std::fwrite(dnext.data(), sizeof(double), P, o) != (size_t)P ||
        std::fwrite(dpos.data(), sizeof(double), P, o) != (size_t)P) {
        std::fprintf(stderr, "FAIL: write %s\n", argv[2]);
        return 1;
    }
    std::fclose(o);
    std::printf("learn_stream: U=%d stores=%d positions=%lld %s\n", U, S,
                (long long)P,
#ifdef __HIPCC__
                "gpu"
#else
                "cpu"
#endif
    );
    return 0;
}
