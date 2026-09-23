// Babbling on the device: S independent streams advanced in lockstep.
//
//   babble_device IN OUT
//
// The CPU babbler (tests/exp_babble.py) is serial ALONG THE SEQUENCE --
// each byte conditions on the one before it, and no hardware changes
// that. What it does not have to be is serial ACROSS STREAMS: seeds are
// independent, so S of them advance together and the per-step work
// (rows x 256 candidates, gated and normalised) is the parallel part.
//
// This matters for the reason the experiment exists rather than for
// speed on its own. The error bar on a self-training result comes from
// the seed count, and on the CPU each seed costs minutes, so 8 was
// chosen because it was affordable rather than because it was enough.
//
// A WARNING AGAINST THE OBVIOUS SHORTCUT: launching one kernel per byte
// would be launch-bound -- a few thousand useful operations against a
// launch cost of tens of microseconds -- and could run SLOWER than the
// CPU. The whole speedup lives in carrying S streams per launch, which
// is why the loop over steps is inside the kernel and the stream index
// is the block index.
//
// EXECUTION ORDER THIS FILE IS BUILT TO:
//   heavy workload -> device. The per-step drive is rows x 256
//     candidates across S streams, and it never leaves the kernel.
//   device I/O -> host. The store is loaded, packed and written by
//     py/locus/knowledge.py and tools/babble_gpu.py, because a second
//     parser in C++ would be a second definition of what a store is,
//     free to drift with no test noticing.
//   precision -> device first, then host. The kernel emits the
//     QUANTITIES it decided from -- the 256 drive values, the
//     normalising total and the uniform it drew -- for the first
//     `dump` steps, and the host recomputes those exactly and rules on
//     them. The device is fast; the host is the arbiter.
//
// The arithmetic is deliberately the SAME code as scoring: unit_at,
// gate_of and block_sum are lifted from learn_device.cpp unchanged, so
// the generator and the scorer cannot drift into two different models.
// tests/exp_babble.py is the oracle: same store, same seed text, same
// stream, byte-for-byte identical output, or this is wrong.

#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// DERIVED, not chosen: one thread per candidate byte, so B is
// BYTE_UNITS (py/locus/encode.py) and the block reduction runs over
// exactly the alphabet being scored. Changing the alphabet changes
// this; nothing else does.
#define B 256
// A BOUND on how many encoder tables fit in the struct passed by value
// to the kernel, matching learn_device.cpp. The encoder is refused
// rather than truncated if it ever exceeds this.
#define MAXT 8
#define HD __host__ __device__

struct Table {
    int32_t k;
    uint32_t seed;
    int32_t off, size;
};

struct Enc {
    int32_t U, T, top;
    Table t[MAXT];
};

// One sparse store: CSR rows over units, plus whether it holds only the
// top-order slot (the hippocampal tier does).
struct Rows {
    const int64_t *rowptr;
    const uint8_t *byte;
    const double *weight;
    int32_t top_only;
};

HD inline int32_t unit_at(const Enc &e, const uint8_t *d, int64_t g,
                          int64_t p, int j)
{
    if (j == 0)
        return d[g];
    const Table &t = e.t[j - 1];
    if (p < t.k - 1)
        return -1;
    uint32_t h = 2166136261u ^ t.seed;
    for (int i = 0; i < t.k; ++i) {
        h ^= d[g - t.k + 1 + i];
        h *= 16777619u;
    }
    return t.off + (int32_t)(h % (uint32_t)t.size);
}

HD inline bool slot_is_top(const Enc &e, int j)
{
    return j > 0 && e.t[j - 1].k == e.top;
}

HD inline double gate_of(double dot_rd, double n2r, double n2d)
{
    double den = sqrt(n2r) * sqrt(n2d - 2.0 * dot_rd + n2r);
    double c = (n2r > 0.0 && den > 0.0) ? (dot_rd - n2r) / den : 0.0;
    double sc = c >= 0.0 ? sqrt(c) : -sqrt(-c);
    return 1.0 / (1.0 + exp(-sc));
}

__device__ double block_sum(double v, double *sh)
{
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int s = B / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s)
            sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    double t = sh[0];
    __syncthreads();
    return t;
}

// A sparse row scattered into a dense 256 vector the block can read.
__device__ void load_row(const Rows &r, int32_t u, double *dst)
{
    dst[threadIdx.x] = 0.0;
    __syncthreads();
    int64_t lo = r.rowptr[u], hi = r.rowptr[u + 1];
    for (int64_t i = lo + threadIdx.x; i < hi; i += B)
        dst[r.byte[i]] = r.weight[i];
    __syncthreads();
}

// splitmix64: a stream's randomness must depend on its own seed alone,
// never on scheduling order, or two runs of the same seed differ.
__device__ inline uint64_t mix64(uint64_t &s)
{
    s += 0x9E3779B97F4A7C15ull;
    uint64_t z = s;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

__device__ inline double next_unit(uint64_t &s)
{
    return (double)(mix64(s) >> 11) * (1.0 / 9007199254740992.0);
}

// One block per stream. `buf` holds seed text followed by room for
// `steps` generated bytes, stride `stride` per stream.
__global__ void babble(Enc e, Rows cortex, Rows hippo, int has_hippo,
                       int gated, double lam, uint8_t *buf, int64_t stride,
                       int64_t seed_len, int64_t steps,
                       const uint64_t *seeds, int32_t dump, double *dbg)
{
    __shared__ double sh[B];
    __shared__ double row[B];
    __shared__ double drive[B];
    __shared__ int picked;

    int64_t s = blockIdx.x;
    int b = threadIdx.x;
    uint8_t *d = buf + s * stride;
    uint64_t rng = seeds[s];

    for (int64_t step = 0; step < steps; ++step) {
        int64_t g = seed_len + step - 1;   // predict the byte after g
        int64_t p = g;                     // one file per stream

        double v = 0.0;
        for (int j = 0; j <= e.T; ++j) {
            int32_t u = unit_at(e, d, g, p, j);
            if (u < 0)
                continue;
            load_row(cortex, u, row);
            v += row[b];
            if (has_hippo && slot_is_top(e, j)) {
                load_row(hippo, u, row);
                v += row[b];
            }
        }

        if (gated) {
            double n2d = block_sum(v * v, sh);
            double gv = 0.0;
            int rows = 0;
            for (int j = 0; j <= e.T; ++j) {
                int32_t u = unit_at(e, d, g, p, j);
                if (u < 0)
                    continue;
                load_row(cortex, u, row);
                double r = row[b];
                double dr = block_sum(r * v, sh);
                double n2r = block_sum(r * r, sh);
                if (n2r > 0.0)
                    ++rows;
                gv += gate_of(dr, n2r, n2d) * r;
                if (has_hippo && slot_is_top(e, j)) {
                    load_row(hippo, u, row);
                    double rh = row[b];
                    double dh = block_sum(rh * v, sh);
                    double n2h = block_sum(rh * rh, sh);
                    if (n2h > 0.0)
                        ++rows;
                    gv += gate_of(dh, n2h, n2d) * rh;
                }
            }
            if (rows > 1)
                v = gv;
        }

        // The reader charges (drive + lam/256) / (sum + lam); sampling
        // reads the same distribution or the babble measures a second,
        // unrelated model.
        drive[b] = v > 0.0 ? v : 0.0;
        __syncthreads();
        double total = block_sum(drive[b], sh) + lam;

        // Precision hand-off: what the device decided FROM, so the host
        // can re-derive it rather than judge the byte alone. Stream 0
        // only -- one stream localises a disagreement, S streams just
        // repeat it.
        if (dbg && s == 0 && step < dump) {
            dbg[step * (B + 2) + b] = drive[b];
            if (b == 0) {
                dbg[step * (B + 2) + B] = total;
            }
        }
        __syncthreads();

        if (b == 0) {
            double u = next_unit(rng);
            if (dbg && s == 0 && step < dump)
                dbg[step * (B + 2) + B + 1] = u;
            double r = u * total;
            double acc = 0.0;
            int pick = B - 1;
            for (int i = 0; i < B; ++i) {
                acc += drive[i] + lam / (double)B;
                if (acc >= r) {
                    pick = i;
                    break;
                }
            }
            picked = pick;
            d[seed_len + step] = (uint8_t)pick;
        }
        __syncthreads();
        (void)picked;
    }
}

static void *slurp(const char *path, size_t *len)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path);
        exit(2);
    }
    fseek(f, 0, SEEK_END);
    *len = (size_t)ftell(f);
    fseek(f, 0, SEEK_SET);
    void *p = malloc(*len);
    if (fread(p, 1, *len, f) != *len) {
        fprintf(stderr, "short read on %s\n", path);
        exit(2);
    }
    fclose(f);
    return p;
}

#define CK(x) do { hipError_t _e = (x); if (_e != hipSuccess) { \
    fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, \
            hipGetErrorString(_e)); exit(3); } } while (0)

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "babble_device IN OUT\n");
        return 2;
    }
    size_t n = 0;
    uint8_t *in = (uint8_t *)slurp(argv[1], &n);
    size_t o = 0;
    auto take = [&](size_t bytes) { void *p = in + o; o += bytes; return p; };
    // Every double array here is read by CASTING INTO THIS BUFFER, so it
    // has to start on an 8-byte boundary. The packer pads after each
    // byte array; skipping that padding is this side of the same
    // contract, and getting it wrong does not fail loudly -- it reads
    // plausible-looking rows with garbage weights (measured: drive
    // values of 1e258 beside a correct 6.44) and a seed list that makes
    // every stream draw different numbers from the host.
    auto align8 = [&]() { o = (o + 7) & ~(size_t)7; };

    Enc e;
    memcpy(&e, take(sizeof(int32_t) * 3), sizeof(int32_t) * 3);
    memcpy(e.t, take(sizeof(Table) * e.T), sizeof(Table) * e.T);
    int32_t gated = *(int32_t *)take(sizeof(int32_t));
    int32_t has_hippo = *(int32_t *)take(sizeof(int32_t));
    int32_t streams = *(int32_t *)take(sizeof(int32_t));
    int32_t dump = *(int32_t *)take(sizeof(int32_t));
    (void)*(int32_t *)take(sizeof(int32_t));   // pad: the double follows
    double lam = *(double *)take(sizeof(double));
    int64_t seed_len = *(int64_t *)take(sizeof(int64_t));
    int64_t steps = *(int64_t *)take(sizeof(int64_t));

    int64_t nnz_c = *(int64_t *)take(sizeof(int64_t));
    int64_t *rp_c = (int64_t *)take(sizeof(int64_t) * (e.U + 1));
    uint8_t *by_c = (uint8_t *)take(nnz_c);
    align8();
    double *w_c = (double *)take(sizeof(double) * nnz_c);

    int64_t nnz_h = 0;
    int64_t *rp_h = nullptr;
    uint8_t *by_h = nullptr;
    double *w_h = nullptr;
    if (has_hippo) {
        nnz_h = *(int64_t *)take(sizeof(int64_t));
        rp_h = (int64_t *)take(sizeof(int64_t) * (e.U + 1));
        by_h = (uint8_t *)take(nnz_h);
        align8();
        w_h = (double *)take(sizeof(double) * nnz_h);
    }
    uint8_t *seed_text = (uint8_t *)take(seed_len);
    align8();
    uint64_t *seeds = (uint64_t *)take(sizeof(uint64_t) * streams);

    int64_t stride = seed_len + steps;
    std::vector<uint8_t> host((size_t)stride * streams, 0);
    for (int s = 0; s < streams; ++s)
        memcpy(&host[(size_t)s * stride], seed_text, seed_len);

    uint8_t *d_buf = nullptr, *d_by_c = nullptr, *d_by_h = nullptr;
    int64_t *d_rp_c = nullptr, *d_rp_h = nullptr;
    double *d_w_c = nullptr, *d_w_h = nullptr;
    uint64_t *d_seeds = nullptr;

    CK(hipMalloc(&d_buf, host.size()));
    CK(hipMemcpy(d_buf, host.data(), host.size(), hipMemcpyHostToDevice));
    CK(hipMalloc(&d_rp_c, sizeof(int64_t) * (e.U + 1)));
    CK(hipMemcpy(d_rp_c, rp_c, sizeof(int64_t) * (e.U + 1),
                 hipMemcpyHostToDevice));
    CK(hipMalloc(&d_by_c, nnz_c ? nnz_c : 1));
    CK(hipMemcpy(d_by_c, by_c, nnz_c, hipMemcpyHostToDevice));
    CK(hipMalloc(&d_w_c, sizeof(double) * (nnz_c ? nnz_c : 1)));
    CK(hipMemcpy(d_w_c, w_c, sizeof(double) * nnz_c, hipMemcpyHostToDevice));
    CK(hipMalloc(&d_seeds, sizeof(uint64_t) * streams));
    CK(hipMemcpy(d_seeds, seeds, sizeof(uint64_t) * streams,
                 hipMemcpyHostToDevice));

    Rows cortex = {d_rp_c, d_by_c, d_w_c, 0};
    Rows hippo = {nullptr, nullptr, nullptr, 1};
    if (has_hippo) {
        CK(hipMalloc(&d_rp_h, sizeof(int64_t) * (e.U + 1)));
        CK(hipMemcpy(d_rp_h, rp_h, sizeof(int64_t) * (e.U + 1),
                     hipMemcpyHostToDevice));
        CK(hipMalloc(&d_by_h, nnz_h ? nnz_h : 1));
        CK(hipMemcpy(d_by_h, by_h, nnz_h, hipMemcpyHostToDevice));
        CK(hipMalloc(&d_w_h, sizeof(double) * (nnz_h ? nnz_h : 1)));
        CK(hipMemcpy(d_w_h, w_h, sizeof(double) * nnz_h,
                     hipMemcpyHostToDevice));
        hippo = {d_rp_h, d_by_h, d_w_h, 1};
    }

    double *d_dbg = nullptr;
    size_t dbg_n = (size_t)dump * (B + 2);
    if (dump > 0)
        CK(hipMalloc(&d_dbg, sizeof(double) * dbg_n));

    hipLaunchKernelGGL(babble, dim3(streams), dim3(B), 0, 0, e, cortex,
                       hippo, has_hippo, gated, lam, d_buf, stride,
                       seed_len, steps, d_seeds, dump, d_dbg);
    CK(hipGetLastError());
    CK(hipDeviceSynchronize());
    CK(hipMemcpy(host.data(), d_buf, host.size(), hipMemcpyDeviceToHost));

    FILE *f = fopen(argv[2], "wb");
    if (!f) {
        fprintf(stderr, "cannot write %s\n", argv[2]);
        return 2;
    }
    for (int s = 0; s < streams; ++s)
        fwrite(&host[(size_t)s * stride + seed_len], 1, steps, f);
    fclose(f);

    if (dump > 0) {
        std::vector<double> dbg(dbg_n);
        CK(hipMemcpy(dbg.data(), d_dbg, sizeof(double) * dbg_n,
                     hipMemcpyDeviceToHost));
        char name[4096];
        snprintf(name, sizeof(name), "%s.dbg", argv[2]);
        FILE *g = fopen(name, "wb");
        if (g) {
            fwrite(dbg.data(), sizeof(double), dbg_n, g);
            fclose(g);
        }
    }
    fprintf(stderr, "babble_device: %d streams x %lld bytes\n", streams,
            (long long)steps);
    return 0;
}
