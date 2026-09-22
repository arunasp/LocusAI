// A whole stream learner on the device: encode, emit events, sort them by
// unit, learn, score and fit lam. The host sends raw bytes, file bounds,
// the encoder's tables, a schedule of training positions and a lam grid.
//
// Mirrors core/py/locus/encode.py (FNV-1a units) and core/py/locus/learn.py
// (per-input delta rule; metaplastic rate 1/n from an event's rank in its
// unit). The control (kind 2) mirrors exp_learn_stream.control_parts:
// count rows, each divided by its unit's total, summed per byte. Builds as
// HIP, or without hipcc as plain C++ with std::stable_sort in place of the
// device radix sort.
//
// Usage: learn_device IN OUT [KNOW]
//   IN  int32 U, int32 T, double rate, int32 kind (0 branch, 1 cls,
//       2 control), int32 flags (1 metaplastic, 2 hippocampus,
//       4 cortex metaplastic, 8 coincidence-gated readout), int32 top;
//       per table: int32 k, uint32 seed, int32 offset, int32 size;
//       train: int64 nbytes, bytes, int32 nfiles, int64 start[nfiles+1];
//       int64 nsched, int64 sched[nsched] (main-store positions in order);
//       int64 nonline, int64 online[nonline] (hippocampus positions);
//       scored: int64 nbytes, bytes, int32 nfiles, int64 start[nfiles+1],
//       int32 nval_files; int32 nlam, double lam[nlam]
//   OUT double val[nlam], double test[nlam], int64 nval, int64 ntest
//   KNOW (optional, learners only) the learned stores, sparse: char[8]
//       "LOCUSKN1", int32 U, int32 nstores, double rate; per store (the
//       neocortex or branch store, then the hippocampus): int32 flags
//       (1 metaplastic, 2 top-order units only), int64 events[U],
//       int64 rowptr[U+1], uint8 byte[nnz], double w[nnz]. A weight never
//       moved stays exactly 0.0, so only non-zero weights are kept.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
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
static const int MAXT = 8;

struct Table {
    int32_t k;
    uint32_t seed;
    int32_t off, size;
};

struct Enc {
    int32_t U, T, top;
    Table t[MAXT];
};

HD inline int64_t file_of(const int64_t *start, int32_t nf, int64_t g)
{
    int32_t lo = 0, hi = nf - 1;
    while (lo < hi) {
        int32_t mid = (lo + hi + 1) / 2;
        if (start[mid] <= g)
            lo = mid;
        else
            hi = mid - 1;
    }
    return lo;
}

// Unit of slot j (0 = the byte, j >= 1 = table j - 1) at global position
// g, whose offset within its file is p; -1 when the n-gram does not fit.
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

// Coincidence gate (tests/exp_match_gate.py): a row counts in proportion
// to its agreement with the sum of the other active rows.
// cos(r, D - r) from dot(r, D), |r|^2 and |D|^2.
HD inline double gate_of(double dot_rd, double n2r, double n2d)
{
    double den = sqrt(n2r) * sqrt(n2d - 2.0 * dot_rd + n2r);
    double c = (n2r > 0.0 && den > 0.0) ? (dot_rd - n2r) / den : 0.0;
    double sc = c >= 0.0 ? sqrt(c) : -sqrt(-c);
    return 1.0 / (1.0 + exp(-sc));
}

HD inline double step(double w, int b, int nxt, double rate, double loc,
                      bool use_loc)
{
    double err = (b == nxt ? 1.0 : 0.0) - w;
    if (err == 0.0)
        return w;
    double dw = rate * err;
    dw = dw * 1.0;
    if (use_loc)
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
// ---------------------------------------------------------------- device --
__global__ void emit(Enc e, const uint8_t *d, const int64_t *fs, int32_t nf,
                     const int64_t *sched, int64_t n, int top_only,
                     uint32_t *key, uint8_t *val)
{
    int64_t s = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (s >= n)
        return;
    int64_t g = sched[s];
    int64_t p = g - fs[file_of(fs, nf, g)];
    int K = e.T + 1;
    for (int j = 0; j < K; ++j) {
        int32_t u = (top_only && !slot_is_top(e, j)) ? -1
                                                     : unit_at(e, d, g, p, j);
        key[s * K + j] = u < 0 ? (uint32_t)e.U : (uint32_t)u;
        val[s * K + j] = d[g + 1];
    }
}

static const int C = 256;

// Events applied per unit per `learn` launch. It bounds how long one
// launch can hold the device, which is what keeps the driver watchdog
// out of this; it does not change any result, only how the same work
// is divided across launches.
static const int64_t LEARN_SPAN = 1 << 20;

// Events each unit has already seen, so the metaplastic 1/n keeps
// counting across batches.
__global__ void add_rows(int64_t *base, const int64_t *rowptr, int32_t U)
{
    int64_t u = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (u < (int64_t)U)
        base[u] += rowptr[u + 1] - rowptr[u];
}

__global__ void rs_count(const uint32_t *key, int64_t n, int shift,
                         int64_t *hist, int64_t nt)
{
    int64_t t = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (t >= nt)
        return;
    int64_t c[256];
    for (int d = 0; d < 256; ++d)
        c[d] = 0;
    int64_t lo = t * C, hi = lo + C < n ? lo + C : n;
    for (int64_t i = lo; i < hi; ++i)
        ++c[(key[i] >> shift) & 255];
    for (int d = 0; d < 256; ++d)
        hist[d * nt + t] = c[d];
}

__global__ void rs_scatter(const uint32_t *kin, const uint8_t *vin,
                           uint32_t *kout, uint8_t *vout, int64_t n,
                           int shift, const int64_t *off, int64_t nt)
{
    int64_t t = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (t >= nt)
        return;
    int64_t o[256];
    for (int d = 0; d < 256; ++d)
        o[d] = off[d * nt + t];
    int64_t lo = t * C, hi = lo + C < n ? lo + C : n;
    for (int64_t i = lo; i < hi; ++i) {
        int d = (kin[i] >> shift) & 255;
        kout[o[d]] = kin[i];
        vout[o[d]] = vin[i];
        ++o[d];
    }
}

__global__ void scan_blk(int64_t *a, int64_t m, int64_t *sums)
{
    __shared__ int64_t s[1024];
    int tid = threadIdx.x;
    int64_t i = blockIdx.x * (int64_t)1024 + tid;
    int64_t v = i < m ? a[i] : 0;
    s[tid] = v;
    __syncthreads();
    for (int off = 1; off < 1024; off <<= 1) {
        int64_t x = tid >= off ? s[tid - off] : 0;
        __syncthreads();
        s[tid] += x;
        __syncthreads();
    }
    if (i < m)
        a[i] = s[tid] - v;
    if (tid == 1023)
        sums[blockIdx.x] = s[1023];
}

__global__ void add_blk(int64_t *a, int64_t m, const int64_t *sums)
{
    int64_t i = blockIdx.x * (int64_t)1024 + threadIdx.x;
    if (i < m)
        a[i] += sums[blockIdx.x];
}

static void scan(int64_t *a, int64_t m)
{
    int64_t nb = (m + 1023) / 1024;
    int64_t *sums;
    CHECK(hipMalloc(&sums, nb * sizeof(int64_t)));
    scan_blk<<<nb, 1024>>>(a, m, sums);
    CHECK(hipGetLastError());
    if (nb > 1) {
        scan(sums, nb);
        add_blk<<<nb, 1024>>>(a, m, sums);
        CHECK(hipGetLastError());
    }
    CHECK(hipFree(sums));
}

__global__ void unit_hist(const uint32_t *key, int64_t n, int64_t *cnt)
{
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i < n)
        atomicAdd((unsigned long long *)&cnt[key[i]], 1ULL);
}

// A unit's events must be applied IN ORDER (the metaplastic rate is
// 1/n over that unit's own history), so one block owns one unit and
// walks its list serially. Event counts per unit are extremely skewed
// in text -- the space character and the commonest bigrams carry a
// large share -- so on a big corpus the hottest block alone would run
// for minutes inside ONE launch, past any driver watchdog: a 134 MB
// corpus hung the device with the card idle at 3% and
// hipDeviceSynchronize never returning.
//
// The fix keeps the arithmetic identical and bounds the LAUNCH: each
// call advances every unit by at most `span` of its own events,
// resuming the accumulator from W (zeroed before the first chunk).
// Same operations, same order, one launch per span.
__global__ void learn(const int64_t *rowptr, const uint8_t *nxt, double rate,
                      int meta, double *W, int64_t chunk_lo, int64_t span,
                      const int64_t *base)
{
    int u = blockIdx.x, b = threadIdx.x;
    int64_t r0 = rowptr[u], r1 = rowptr[u + 1];
    int64_t lo = r0 + chunk_lo;
    if (lo >= r1)
        return;
    int64_t hi = lo + span < r1 ? lo + span : r1;
    double w = W[(int64_t)u * B + b];
    for (int64_t e = lo; e < hi; ++e) {
        double loc = meta
                         ? 1.0 / (rate * (double)(base[u] + (e - r0) + 1))
                         : 1.0;
        w = step(w, b, nxt[e], rate, loc, meta != 0);
    }
    W[(int64_t)u * B + b] = w;
}

__global__ void count_rows(Enc e, const uint8_t *d, const int64_t *fs,
                           int32_t nf, const int64_t *sched, int64_t n,
                           unsigned int *cnt, unsigned long long *tot)
{
    int64_t s = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (s >= n)
        return;
    int64_t g = sched[s];
    int64_t p = g - fs[file_of(fs, nf, g)];
    for (int j = 0; j <= e.T; ++j) {
        int32_t u = unit_at(e, d, g, p, j);
        if (u < 0)
            continue;
        atomicAdd(&cnt[(int64_t)u * B + d[g + 1]], 1u);
        atomicAdd(&tot[u], 1ULL);
    }
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

__global__ void score(Enc e, const uint8_t *d, const int64_t *fs, int32_t nf,
                      const int64_t *pos, int kind, int gated,
                      const double *Wc, const double *Wh,
                      const unsigned int *cnt,
                      const unsigned long long *tot, double *dnext,
                      double *dpos)
{
    __shared__ double sh[B];
    int64_t q = blockIdx.x;
    int b = threadIdx.x;
    int64_t g = pos[q];
    int64_t p = g - fs[file_of(fs, nf, g)];
    double dc = 0.0, dh = 0.0;
    for (int j = 0; j <= e.T; ++j) {
        int32_t u = unit_at(e, d, g, p, j);
        if (u < 0)
            continue;
        if (kind == 2) {
            unsigned long long t = tot[u];
            unsigned int c = cnt[(int64_t)u * B + b];
            if (t && c)
                dc = dc + (double)c / (double)t;
        } else {
            dc = dc + Wc[(int64_t)u * B + b];
            if (Wh && slot_is_top(e, j))
                dh = dh + Wh[(int64_t)u * B + b];
        }
    }
    double v = Wh ? dc + dh : dc;
    if (gated && kind != 2) {
        double n2d = block_sum(v * v, sh);
        double gv = 0.0;
        int rows = 0;
        for (int j = 0; j <= e.T; ++j) {
            int32_t u = unit_at(e, d, g, p, j);
            if (u < 0)
                continue;
            double r = Wc[(int64_t)u * B + b];
            double dr = block_sum(r * v, sh);
            double n2r = block_sum(r * r, sh);
            if (n2r > 0.0)
                ++rows;
            gv = gv + gate_of(dr, n2r, n2d) * r;
            if (Wh && slot_is_top(e, j)) {
                double rh = Wh[(int64_t)u * B + b];
                double dh2 = block_sum(rh * v, sh);
                double n2h = block_sum(rh * rh, sh);
                if (n2h > 0.0)
                    ++rows;
                gv = gv + gate_of(dh2, n2h, n2d) * rh;
            }
        }
        if (rows > 1)
            v = gv;
    }
    sh[b] = v > 0.0 ? v : 0.0;
    __syncthreads();
    if (b == d[g + 1])
        dnext[q] = sh[b];
    for (int s = B / 2; s > 0; s >>= 1) {
        if (b < s)
            sh[b] += sh[b + s];
        __syncthreads();
    }
    if (b == 0)
        dpos[q] = sh[0];
}

__global__ void lam_bits(const double *dnext, const double *dpos,
                         int64_t lo, int64_t hi, const double *lam,
                         double *out)
{
    __shared__ double sh[1024];
    double l = lam[blockIdx.x], acc = 0.0;
    for (int64_t i = lo + threadIdx.x; i < hi; i += 1024)
        acc -= log2((dnext[i] + l / 256.0) / (dpos[i] + l));
    sh[threadIdx.x] = acc;
    __syncthreads();
    for (int s = 512; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s)
            sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        out[blockIdx.x] = sh[0];
}

template <class T>
static T *up(const std::vector<T> &v)
{
    T *p;
    CHECK(hipMalloc(&p, std::max<size_t>(1, v.size()) * sizeof(T)));
    if (!v.empty())
        CHECK(hipMemcpy(p, v.data(), v.size() * sizeof(T),
                        hipMemcpyHostToDevice));
    return p;
}

// One store on the device: emit, sort, offsets, learn. Returns W.
static double *store(const Enc &e, const uint8_t *dd, const int64_t *dfs,
                     int32_t nf, const std::vector<int64_t> &sched,
                     bool top_only, bool meta, double rate,
                     std::vector<int64_t> *events)
{
    int64_t n = sched.size(), K = e.T + 1;

    // Peak VRAM here is LINEAR IN EVENTS: 4+4 bytes of radix keys, 1+1
    // of values and 8 of histogram per event, about 18 B. A 134 MB
    // corpus (~630M events) therefore asked for ~11 GB and, with the
    // weight arrays, left a 20 GB card at 19.8 GB in use -- nothing
    // spare for the desktop, and the driver hung with the card idle,
    // the same failure a large model causes when it leaves no headroom.
    //
    // So the schedule runs in BATCHES sized from the free memory
    // measured now, minus a reserve for the rest of the system
    // (LOCUS_VRAM_RESERVE_MB, default 3072 -- an initial value from
    // Arunas's experience of Ollama needing ~3 GB free, not a tuned
    // constant). Results are unchanged: batches follow stream order, so
    // every unit still sees its own events in order, and `base` carries
    // how many it has seen so the metaplastic 1/n keeps counting.
    size_t freeb = 0, totalb = 0;
    CHECK(hipMemGetInfo(&freeb, &totalb));
    const char *rv = std::getenv("LOCUS_VRAM_RESERVE_MB");
    size_t reserve = (size_t)(rv ? std::atoll(rv) : 3072) << 20;
    size_t weights = (size_t)e.U * B * sizeof(double) + (size_t)e.U * 8;
    size_t usable = freeb > reserve + weights ? freeb - reserve - weights
                                              : 0;
    int64_t per_event = 4 + 4 + 1 + 1 + 8;
    int64_t max_events = std::max<int64_t>(
        1 << 20, (int64_t)(usable / (size_t)per_event));
    int64_t batch = std::max<int64_t>(1, max_events / K);
    if (n && batch < n)
        std::fprintf(stderr,
                     "learn_device: %lld positions in batches of %lld "
                     "(%.1f GB free, %.1f GB reserved)\n",
                     (long long)n, (long long)batch, freeb / 1e9,
                     (double)reserve / 1e9);

    double *W;
    CHECK(hipMalloc(&W, (size_t)e.U * B * sizeof(double)));
    CHECK(hipMemset(W, 0, (size_t)e.U * B * sizeof(double)));
    int64_t *base;
    CHECK(hipMalloc(&base, std::max<size_t>(1, (size_t)e.U) * 8));
    CHECK(hipMemset(base, 0, std::max<size_t>(1, (size_t)e.U) * 8));

    for (int64_t p0 = 0; p0 < n; p0 += batch) {
        int64_t bn = std::min<int64_t>(batch, n - p0);
        int64_t E = bn * K;
        std::vector<int64_t> slice(sched.begin() + p0,
                                   sched.begin() + p0 + bn);
        int64_t *ds = up(slice);
        uint32_t *k0, *k1;
        uint8_t *v0, *v1;
        CHECK(hipMalloc(&k0, (E + 1) * 4));
        CHECK(hipMalloc(&k1, (E + 1) * 4));
        CHECK(hipMalloc(&v0, E + 1));
        CHECK(hipMalloc(&v1, E + 1));
        emit<<<(bn + 255) / 256, 256>>>(e, dd, dfs, nf, ds, bn, top_only,
                                        k0, v0);
        CHECK(hipGetLastError());
        int64_t nt = (E + C - 1) / C;
        int64_t *hist;
        CHECK(hipMalloc(&hist, std::max<int64_t>(1, 256 * nt) * 8));
        for (int shift = 0; shift < 24 && E; shift += 8) {
            rs_count<<<(nt + 255) / 256, 256>>>(k0, E, shift, hist, nt);
            CHECK(hipGetLastError());
            scan(hist, 256 * nt);
            rs_scatter<<<(nt + 255) / 256, 256>>>(k0, v0, k1, v1, E, shift,
                                                  hist, nt);
            CHECK(hipGetLastError());
            std::swap(k0, k1);
            std::swap(v0, v1);
        }
        int64_t *rowptr;
        CHECK(hipMalloc(&rowptr, (e.U + 2) * 8));
        CHECK(hipMemset(rowptr, 0, (e.U + 2) * 8));
        if (E) {
            unit_hist<<<(E + 255) / 256, 256>>>(k0, E, rowptr);
            CHECK(hipGetLastError());
        }
        scan(rowptr, e.U + 2);
        std::vector<int64_t> rp_host(e.U + 1);
        CHECK(hipMemcpy(rp_host.data(), rowptr, (e.U + 1) * 8,
                        hipMemcpyDeviceToHost));
        int64_t max_row = 0;
        for (int32_t u = 0; u < e.U; ++u)
            max_row = std::max(max_row, rp_host[u + 1] - rp_host[u]);
        for (int64_t lo = 0; lo < max_row; lo += LEARN_SPAN) {
            learn<<<e.U, B>>>(rowptr, v0, rate, meta ? 1 : 0, W, lo,
                              LEARN_SPAN, base);
            CHECK(hipGetLastError());
            CHECK(hipDeviceSynchronize());
        }
        add_rows<<<(e.U + 255) / 256, 256>>>(base, rowptr, e.U);
        CHECK(hipGetLastError());
        CHECK(hipDeviceSynchronize());
        for (void *q : {(void *)ds, (void *)k0, (void *)k1, (void *)v0,
                        (void *)v1, (void *)hist, (void *)rowptr})
            CHECK(hipFree(q));
    }
    if (events) {
        events->assign(e.U, 0);
        if (e.U)
            CHECK(hipMemcpy(events->data(), base, (size_t)e.U * 8,
                            hipMemcpyDeviceToHost));
    }
    CHECK(hipFree(base));
    return W;
}
#else
// ------------------------------------------------------------------ host --
static std::vector<double> store(const Enc &e, const std::vector<uint8_t> &d,
                                 const std::vector<int64_t> &fs,
                                 const std::vector<int64_t> &sched,
                                 bool top_only, bool meta, double rate,
                                 std::vector<int64_t> *events)
{
    int K = e.T + 1;
    std::vector<std::pair<uint32_t, uint8_t>> ev;
    ev.reserve(sched.size() * K);
    for (int64_t g : sched) {
        int64_t p = g - fs[file_of(fs.data(), (int32_t)fs.size() - 1, g)];
        for (int j = 0; j < K; ++j) {
            int32_t u = (top_only && !slot_is_top(e, j))
                            ? -1
                            : unit_at(e, d.data(), g, p, j);
            ev.push_back({u < 0 ? (uint32_t)e.U : (uint32_t)u, d[g + 1]});
        }
    }
    std::stable_sort(ev.begin(), ev.end(),
                     [](const std::pair<uint32_t, uint8_t> &a,
                        const std::pair<uint32_t, uint8_t> &b) {
                         return a.first < b.first;
                     });
    std::vector<double> W((size_t)e.U * B, 0.0);
    if (events)
        events->assign(e.U, 0);
    size_t i = 0;
    while (i < ev.size() && ev[i].first < (uint32_t)e.U) {
        uint32_t u = ev[i].first;
        size_t j = i;
        while (j < ev.size() && ev[j].first == u)
            ++j;
        if (events)
            (*events)[u] = (int64_t)(j - i);
        for (int b = 0; b < B; ++b) {
            double w = 0.0;
            for (size_t x = i; x < j; ++x) {
                double loc = meta ? 1.0 / (rate * (double)(x - i + 1)) : 1.0;
                w = step(w, b, ev[x].second, rate, loc, meta);
            }
            W[(size_t)u * B + b] = w;
        }
        i = j;
    }
    return W;
}
#endif

struct Stored {
    int32_t flags;
    std::vector<int64_t> events;
    std::vector<double> W;
};

static void write_knowledge(const char *path, int32_t U, double rate,
                            const std::vector<Stored> &st)
{
    FILE *o = std::fopen(path, "wb");
    if (!o) {
        std::fprintf(stderr, "FAIL: open %s\n", path);
        std::exit(1);
    }
    int32_t ns = (int32_t)st.size();
    std::fwrite("LOCUSKN1", 1, 8, o);
    std::fwrite(&U, 4, 1, o);
    std::fwrite(&ns, 4, 1, o);
    std::fwrite(&rate, 8, 1, o);
    for (const Stored &s : st) {
        std::vector<int64_t> rp(U + 1, 0);
        std::vector<uint8_t> bytes;
        std::vector<double> w;
        for (int32_t u = 0; u < U; ++u) {
            for (int b = 0; b < B; ++b) {
                double x = s.W[(size_t)u * B + b];
                if (x != 0.0) {
                    bytes.push_back((uint8_t)b);
                    w.push_back(x);
                }
            }
            rp[u + 1] = (int64_t)w.size();
        }
        std::fwrite(&s.flags, 4, 1, o);
        std::fwrite(s.events.data(), 8, U, o);
        std::fwrite(rp.data(), 8, U + 1, o);
        if (!bytes.empty()) {
            std::fwrite(bytes.data(), 1, bytes.size(), o);
            std::fwrite(w.data(), 8, w.size(), o);
        }
    }
    if (std::fclose(o) != 0) {
        std::fprintf(stderr, "FAIL: write %s\n", path);
        std::exit(1);
    }
}

template <class T>
static void rd(FILE *f, T *p, size_t n)
{
    if (n && std::fread(p, sizeof(T), n, f) != n) {
        std::fprintf(stderr, "FAIL: short read\n");
        std::exit(1);
    }
}

template <class T>
static std::vector<T> rdv(FILE *f)
{
    int64_t n;
    rd(f, &n, 1);
    std::vector<T> v(n);
    rd(f, v.data(), n);
    return v;
}

int main(int argc, char **argv)
{
    if (argc != 3 && argc != 4) {
        std::fprintf(stderr, "usage: learn_device IN OUT [KNOW]\n");
        return 2;
    }
    FILE *f = std::fopen(argv[1], "rb");
    if (!f) {
        std::fprintf(stderr, "FAIL: open %s\n", argv[1]);
        return 1;
    }
    Enc e;
    double rate;
    int32_t kind, flags;
    rd(f, &e.U, 1);
    rd(f, &e.T, 1);
    rd(f, &rate, 1);
    rd(f, &kind, 1);
    rd(f, &flags, 1);
    rd(f, &e.top, 1);
    if (e.T > MAXT || e.U >= (1 << 24)) {
        std::fprintf(stderr, "FAIL: T=%d U=%d out of range\n", e.T, e.U);
        return 1;
    }
    for (int i = 0; i < e.T; ++i) {
        rd(f, &e.t[i].k, 1);
        rd(f, &e.t[i].seed, 1);
        rd(f, &e.t[i].off, 1);
        rd(f, &e.t[i].size, 1);
    }
    std::vector<uint8_t> tr = rdv<uint8_t>(f);
    int32_t nft;
    rd(f, &nft, 1);
    std::vector<int64_t> tfs(nft + 1);
    rd(f, tfs.data(), nft + 1);
    std::vector<int64_t> sched = rdv<int64_t>(f);
    std::vector<int64_t> online = rdv<int64_t>(f);
    std::vector<uint8_t> sc = rdv<uint8_t>(f);
    int32_t nfs, nvf;
    rd(f, &nfs, 1);
    std::vector<int64_t> sfs(nfs + 1);
    rd(f, sfs.data(), nfs + 1);
    rd(f, &nvf, 1);
    int32_t nl;
    rd(f, &nl, 1);
    std::vector<double> lam(nl);
    rd(f, lam.data(), nl);
    std::fclose(f);

    // Scored positions: every byte with a next byte in its file, val first.
    std::vector<int64_t> pos;
    int64_t nval = 0;
    for (int i = 0; i < nfs; ++i) {
        for (int64_t g = sfs[i]; g + 1 < sfs[i + 1]; ++g)
            pos.push_back(g);
        if (i + 1 == nvf)
            nval = pos.size();
    }
    if (nvf == 0)
        nval = 0;
    int64_t P = pos.size();
    bool meta = flags & 1, hippo = flags & 2, cmeta = flags & 4;
    bool gated = flags & 8;
    std::vector<double> vb(nl, 0.0), tb(nl, 0.0);
    const char *know = argc == 4 ? argv[3] : nullptr;
    std::vector<Stored> kept;

#ifdef __HIPCC__
    uint8_t *dtr = up(tr), *dsc = up(sc);
    int64_t *dtfs = up(tfs), *dsfs = up(sfs), *dpos = up(pos);
    double *Wc = nullptr, *Wh = nullptr;
    unsigned int *cnt = nullptr;
    unsigned long long *tot = nullptr;
    if (kind == 2) {
        int64_t *don = up(online);
        CHECK(hipMalloc(&cnt, (size_t)e.U * B * 4));
        CHECK(hipMalloc(&tot, (size_t)e.U * 8));
        CHECK(hipMemset(cnt, 0, (size_t)e.U * B * 4));
        CHECK(hipMemset(tot, 0, (size_t)e.U * 8));
        int64_t n = online.size();
        if (n) {
            count_rows<<<(n + 255) / 256, 256>>>(e, dtr, dtfs, nft, don, n,
                                                 cnt, tot);
            CHECK(hipGetLastError());
        }
        CHECK(hipFree(don));
    } else {
        bool cm = kind == 0 ? meta : cmeta;
        std::vector<int64_t> ec, eh;
        Wc = store(e, dtr, dtfs, nft, sched, false, cm, rate, &ec);
        if (kind == 1 && hippo)
            Wh = store(e, dtr, dtfs, nft, online, true, true, rate, &eh);
        if (know) {
            std::vector<double> h((size_t)e.U * B);
            CHECK(hipMemcpy(h.data(), Wc, h.size() * 8,
                            hipMemcpyDeviceToHost));
            kept.push_back({cm ? 1 : 0, ec, h});
            if (Wh) {
                CHECK(hipMemcpy(h.data(), Wh, h.size() * 8,
                                hipMemcpyDeviceToHost));
                kept.push_back({3, eh, h});
            }
        }
    }
    double *dn, *dp, *dl, *dout;
    CHECK(hipMalloc(&dn, std::max<int64_t>(1, P) * 8));
    CHECK(hipMalloc(&dp, std::max<int64_t>(1, P) * 8));
    if (P) {
        score<<<P, B>>>(e, dsc, dsfs, nfs, dpos, kind, gated ? 1 : 0, Wc,
                        Wh, cnt, tot, dn, dp);
        CHECK(hipGetLastError());
    }
    dl = up(lam);
    CHECK(hipMalloc(&dout, nl * 8));
    lam_bits<<<nl, 1024>>>(dn, dp, 0, nval, dl, dout);
    CHECK(hipMemcpy(vb.data(), dout, nl * 8, hipMemcpyDeviceToHost));
    lam_bits<<<nl, 1024>>>(dn, dp, nval, P, dl, dout);
    CHECK(hipMemcpy(tb.data(), dout, nl * 8, hipMemcpyDeviceToHost));
    CHECK(hipDeviceSynchronize());
#else
    std::vector<double> Wc, Wh;
    std::vector<unsigned int> cnt;
    std::vector<unsigned long long> tot;
    if (kind == 2) {
        cnt.assign((size_t)e.U * B, 0);
        tot.assign(e.U, 0);
        for (int64_t g : online) {
            int64_t p = g - tfs[file_of(tfs.data(), nft, g)];
            for (int j = 0; j <= e.T; ++j) {
                int32_t u = unit_at(e, tr.data(), g, p, j);
                if (u < 0)
                    continue;
                ++cnt[(size_t)u * B + tr[g + 1]];
                ++tot[u];
            }
        }
    } else {
        bool cm = kind == 0 ? meta : cmeta;
        std::vector<int64_t> ec, eh;
        Wc = store(e, tr, tfs, sched, false, cm, rate, &ec);
        if (kind == 1 && hippo)
            Wh = store(e, tr, tfs, online, true, true, rate, &eh);
        if (know) {
            kept.push_back({cm ? 1 : 0, ec, Wc});
            if (!Wh.empty())
                kept.push_back({3, eh, Wh});
        }
    }
    std::vector<double> dn(P), dp(P);
    std::vector<std::vector<double>> rows;
    std::vector<double> dvec(B);
    for (int64_t q = 0; q < P; ++q) {
        int64_t g = pos[q];
        int64_t p = g - sfs[file_of(sfs.data(), nfs, g)];
        double total = 0.0;
        if (gated && kind != 2) {
            rows.clear();
            for (int j = 0; j <= e.T; ++j) {
                int32_t u = unit_at(e, sc.data(), g, p, j);
                if (u < 0)
                    continue;
                rows.push_back(std::vector<double>(
                    &Wc[(size_t)u * B], &Wc[(size_t)u * B] + B));
                if (!Wh.empty() && slot_is_top(e, j))
                    rows.push_back(std::vector<double>(
                        &Wh[(size_t)u * B], &Wh[(size_t)u * B] + B));
            }
            for (int b = 0; b < B; ++b) {
                dvec[b] = 0.0;
                for (const std::vector<double> &r : rows)
                    dvec[b] += r[b];
            }
            double n2d = 0.0;
            for (int b = 0; b < B; ++b)
                n2d += dvec[b] * dvec[b];
            int live = 0;
            for (const std::vector<double> &r : rows) {
                double n2r = 0.0;
                for (int b = 0; b < B; ++b)
                    n2r += r[b] * r[b];
                if (n2r > 0.0)
                    ++live;
            }
            std::vector<double> gv(B, 0.0);
            for (const std::vector<double> &r : rows) {
                double dr = 0.0, n2r = 0.0;
                for (int b = 0; b < B; ++b) {
                    dr += r[b] * dvec[b];
                    n2r += r[b] * r[b];
                }
                double gg = gate_of(dr, n2r, n2d);
                for (int b = 0; b < B; ++b)
                    gv[b] += gg * r[b];
            }
            for (int b = 0; b < B; ++b) {
                double x = live > 1 ? gv[b] : dvec[b];
                x = x > 0.0 ? x : 0.0;
                if (b == sc[g + 1])
                    dn[q] = x;
                total += x;
            }
            dp[q] = total;
            continue;
        }
        for (int b = 0; b < B; ++b) {
            double dc = 0.0, dh = 0.0;
            for (int j = 0; j <= e.T; ++j) {
                int32_t u = unit_at(e, sc.data(), g, p, j);
                if (u < 0)
                    continue;
                if (kind == 2) {
                    unsigned long long t = tot[u];
                    unsigned int c = cnt[(size_t)u * B + b];
                    if (t && c)
                        dc = dc + (double)c / (double)t;
                } else {
                    dc = dc + Wc[(size_t)u * B + b];
                    if (!Wh.empty() && slot_is_top(e, j))
                        dh = dh + Wh[(size_t)u * B + b];
                }
            }
            double v = !Wh.empty() ? dc + dh : dc;
            v = v > 0.0 ? v : 0.0;
            if (b == sc[g + 1])
                dn[q] = v;
            total += v;
        }
        dp[q] = total;
    }
    for (int l = 0; l < nl; ++l)
        for (int64_t q = 0; q < P; ++q) {
            double x = -std::log2((dn[q] + lam[l] / 256.0) / (dp[q] + lam[l]));
            (q < nval ? vb[l] : tb[l]) += x;
        }
#endif
    int64_t nt = P - nval;
    for (int l = 0; l < nl; ++l) {
        vb[l] /= std::max<int64_t>(nval, 1);
        tb[l] /= std::max<int64_t>(nt, 1);
    }
    if (know && kind != 2)
        write_knowledge(know, e.U, rate, kept);
    FILE *o = std::fopen(argv[2], "wb");
    if (!o || std::fwrite(vb.data(), 8, nl, o) != (size_t)nl ||
        std::fwrite(tb.data(), 8, nl, o) != (size_t)nl ||
        std::fwrite(&nval, 8, 1, o) != 1 || std::fwrite(&nt, 8, 1, o) != 1) {
        std::fprintf(stderr, "FAIL: write %s\n", argv[2]);
        return 1;
    }
    std::fclose(o);
    return 0;
}
