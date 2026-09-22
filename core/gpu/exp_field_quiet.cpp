// Quiet rows, a source-row layout, and per-step cost below the launch
// floor, for Field.step on the GPU.
//
// Compiled with FMA contraction off: every product is rounded before its
// add, which the exactness argument below needs.
//
// Quiet: in round-to-nearest, x + y == x whenever 0 <= y < ulp(x) / 2. A
// term a[i] * w below half an ulp of the running drive therefore changes
// nothing, so skipping it is exact. A row is quiet for a step when every
// one of its terms is skipped. The threshold is set by fp32, not chosen.
// Control: skipping below 2 ulp must change the trajectory.
//
// Two-stage: stage 1 runs per SOURCE row and writes each term to an edge
// buffer; stage 2 gathers per target in ascending source order. The source
// row becomes the unit, as residency needs, and the result must be
// bit-identical to the gather kernel. Control: descending gather differs.
//
// Timing: batches of back-to-back steps, so the per-step figure excludes
// host launch latency. Sampled until stable.
#pragma clang fp contract(off)
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
static const float FLOOR = 1e-12f;

struct Params {
    int n, areas;
    float rho, g, beta, leak, dt, amax;
};

__device__ void block_partials(float v, int j, int n, int areas, float *sh, float *part)
{
    for (int k = 0; k < areas; ++k) {
        sh[threadIdx.x] = (j < n && j % areas == k) ? v : 0.0f;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if ((int)threadIdx.x < s)
                sh[threadIdx.x] += sh[threadIdx.x + s];
            __syncthreads();
        }
        if (threadIdx.x == 0)
            part[blockIdx.x * areas + k] = sh[0];
        __syncthreads();
    }
}

__global__ void reduce_kernel(const float *a, float *part, Params p)
{
    __shared__ float sh[BD];
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    block_partials(j < p.n ? a[j] : 0.0f, j, p.n, p.areas, sh, part);
}

__device__ inline float finish(float aj, float drive, int j, const float *part_in, Params p)
{
    int area = j % p.areas;
    float total = 0.0f;
    for (int b = 0; b < (int)gridDim.x; ++b)
        total += part_in[b * p.areas + area];
    float num = p.rho * aj;
    num = num + p.g * drive;
    float net = num / (1.0f + p.beta * (total - aj));
    float v = aj + p.dt * (net - p.leak * aj);
    return v < 0.0f ? 0.0f : (v > p.amax ? p.amax : v);
}

// skip_ulps < 0: never skip. Otherwise skip a term below skip_ulps * ulp of
// the running drive, count it, and mark the source row used when it is not
// skipped. Counters never feed back into the values.
__global__ void gather_step(const int *__restrict__ colptr, const int *__restrict__ src,
                            const float *__restrict__ w, const float *__restrict__ a,
                            float *__restrict__ nxt, const float *__restrict__ part_in,
                            float *__restrict__ part_out, Params p, float skip_ulps,
                            unsigned long long *skipped, int *row_used)
{
    __shared__ float sh[BD];
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    float v = 0.0f;
    if (j < p.n) {
        float drive = 0.0f;
        unsigned long long sk = 0;
        for (int k = colptr[j]; k < colptr[j + 1]; ++k) {
            int i = src[k];
            float ai = a[i];
            if (!(ai > FLOOR))
                continue;
            float term = ai * w[k];
            if (skip_ulps >= 0.0f && drive > 0.0f) {
                float ulp = nextafterf(drive, INFINITY) - drive;
                if (term < skip_ulps * ulp) {
                    ++sk;
                    continue;
                }
            }
            if (row_used)
                atomicOr(&row_used[i], 1);
            drive = drive + term;
        }
        if (skipped && sk)
            atomicAdd(skipped, sk);
        v = finish(a[j], drive, j, part_in, p);
        nxt[j] = v;
    }
    block_partials(v, j, p.n, p.areas, sh, part_out);
}

// Stage 1: per source row, write every outgoing term to the edge buffer.
__global__ void terms_by_source(const int *__restrict__ rowptr, const float *__restrict__ w_row,
                                const float *__restrict__ a, float *__restrict__ term, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    float ai = a[i];
    bool on = ai > FLOOR;
    for (int e = rowptr[i]; e < rowptr[i + 1]; ++e)
        term[e] = on ? ai * w_row[e] : -1.0f;  // -1 marks a skipped source
}

// Stage 2: per target, add its incoming terms in ascending source order.
__global__ void gather_terms(const int *__restrict__ colptr, const int *__restrict__ edge,
                             const float *__restrict__ term, const float *__restrict__ a,
                             float *__restrict__ nxt, const float *__restrict__ part_in,
                             float *__restrict__ part_out, Params p)
{
    __shared__ float sh[BD];
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    float v = 0.0f;
    if (j < p.n) {
        float drive = 0.0f;
        for (int k = colptr[j]; k < colptr[j + 1]; ++k) {
            float t = term[edge[k]];
            if (t >= 0.0f)
                drive = drive + t;
        }
        v = finish(a[j], drive, j, part_in, p);
        nxt[j] = v;
    }
    block_partials(v, j, p.n, p.areas, sh, part_out);
}

__global__ void inject_kernel(float *a, float *part, int idx, float amt, Params p)
{
    __shared__ float sh[BD];
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j == idx)
        a[j] = fminf(p.amax, a[j] + amt);
    __syncthreads();
    block_partials(j < p.n ? a[j] : 0.0f, j, p.n, p.areas, sh, part);
}

// ---------------------------------------------------------------- host --

static std::vector<float> ring_kernel(int n, uint32_t seed, int reach = 3)
{
    auto rnd = [&seed] {
        seed = seed * 1664525u + 1013904223u;
        return (seed >> 8) / 16777216.0f;
    };
    std::vector<float> K((size_t)n * n, 0.0f);
    for (int i = 0; i < n; ++i) {
        float *row = &K[(size_t)i * n];
        row[i] = 0.4f + 0.3f * rnd();
        for (int d = 1; d <= reach; ++d) {
            float wt = std::pow(0.6f, (float)d) * (0.4f + rnd());
            row[(i + d) % n] += wt;
            row[(i - d + n) % n] += wt;
        }
        float s = 0.0f;
        for (int k = 0; k < n; ++k)
            s += row[k];
        for (int k = 0; k < n; ++k)
            row[k] /= s;
    }
    return K;
}

// Both layouts over one edge numbering: edges are numbered by source row
// (CSR order), and each target lists its incoming edge ids by source.
struct Graph {
    std::vector<int> rowptr, dst;          // CSR by source
    std::vector<float> w_row;              // weight per edge, CSR order
    std::vector<int> colptr, src, edge;    // per target: sources, edge ids
    std::vector<float> w_col;              // weight per incoming entry
};

static Graph build(const std::vector<float> &K, int n, bool descending)
{
    Graph g;
    g.rowptr.push_back(0);
    std::vector<std::vector<std::pair<int, int>>> in(n);  // target -> (source, edge)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            float x = K[(size_t)i * n + j];
            if (x != 0.0f) {
                in[j].push_back({i, (int)g.dst.size()});
                g.dst.push_back(j);
                g.w_row.push_back(x);
            }
        }
        g.rowptr.push_back((int)g.dst.size());
    }
    g.colptr.push_back(0);
    for (int j = 0; j < n; ++j) {
        auto &c = in[j];
        if (descending)
            std::reverse(c.begin(), c.end());
        for (auto &e : c) {
            g.src.push_back(e.first);
            g.edge.push_back(e.second);
            g.w_col.push_back(g.w_row[e.second]);
        }
        g.colptr.push_back((int)g.src.size());
    }
    return g;
}

template <class T>
static T *upload(const std::vector<T> &v)
{
    T *d;
    CHECK(hipMalloc(&d, std::max<size_t>(1, v.size()) * sizeof(T)));
    CHECK(hipMemcpy(d, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice));
    return d;
}

enum Kind { GATHER, TWOSTAGE };

struct Device {
    Params p;
    int blocks;
    Kind kind;
    float skip_ulps;
    int *rowptr, *colptr, *src, *edge;
    float *w_row, *w_col, *term;
    float *a, *b, *saved;
    float *part[2];
    unsigned long long *skipped;
    int *row_used;

    Device(const Graph &g, Params p_, Kind k, float skip) : p(p_), kind(k), skip_ulps(skip)
    {
        blocks = (p.n + BD - 1) / BD;
        rowptr = upload(g.rowptr);
        colptr = upload(g.colptr);
        src = upload(g.src);
        edge = upload(g.edge);
        w_row = upload(g.w_row);
        w_col = upload(g.w_col);
        CHECK(hipMalloc(&term, std::max<size_t>(1, g.w_row.size()) * sizeof(float)));
        CHECK(hipMalloc(&a, p.n * sizeof(float)));
        CHECK(hipMalloc(&b, p.n * sizeof(float)));
        CHECK(hipMalloc(&saved, p.n * sizeof(float)));
        for (auto &q : part)
            CHECK(hipMalloc(&q, blocks * p.areas * sizeof(float)));
        CHECK(hipMalloc(&skipped, sizeof(unsigned long long)));
        CHECK(hipMalloc(&row_used, p.n * sizeof(int)));
    }
    ~Device()
    {
        // Teardown: a failure here cannot change any result already printed.
        for (void *q : {(void *)rowptr, (void *)colptr, (void *)src, (void *)edge,
                        (void *)w_row, (void *)w_col, (void *)term, (void *)a, (void *)b,
                        (void *)saved, (void *)part[0], (void *)part[1], (void *)skipped,
                        (void *)row_used})
            (void)hipFree(q);
    }

    void set_state(float level)
    {
        std::vector<float> h(p.n, level);
        CHECK(hipMemcpy(a, h.data(), p.n * sizeof(float), hipMemcpyHostToDevice));
        reduce_kernel<<<blocks, BD>>>(a, part[0], p);
        CHECK(hipGetLastError());
    }

    void step(bool count = false)
    {
        if (kind == GATHER) {
            gather_step<<<blocks, BD>>>(colptr, src, w_col, a, b, part[0], part[1], p,
                                        skip_ulps, count ? skipped : nullptr,
                                        count ? row_used : nullptr);
        } else {
            terms_by_source<<<blocks, BD>>>(rowptr, w_row, a, term, p.n);
            gather_terms<<<blocks, BD>>>(colptr, edge, term, a, b, part[0], part[1], p);
        }
        CHECK(hipGetLastError());
        std::swap(a, b);
        std::swap(part[0], part[1]);
    }

    void inject(int idx, float amt)
    {
        inject_kernel<<<blocks, BD>>>(a, part[0], idx, amt, p);
        CHECK(hipGetLastError());
    }

    std::vector<float> state()
    {
        std::vector<float> h(p.n);
        CHECK(hipMemcpy(h.data(), a, p.n * sizeof(float), hipMemcpyDeviceToHost));
        return h;
    }
};

static void drive_run(Device &d, int steps, int every, uint32_t seed)
{
    d.set_state(0.02f);
    for (long s = 0; s < steps; ++s) {
        if (s % every == 0) {
            seed = seed * 1664525u + 1013904223u;
            d.inject((int)((seed >> 8) % (uint32_t)d.p.n), 0.2f + (seed & 0xFF) / 255.0f);
        }
        d.step();
    }
    CHECK(hipDeviceSynchronize());
}

static bool same(Device &x, Device &y)
{
    auto a = x.state(), b = y.state();
    return std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
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
    std::vector<double> x(s.begin(), s.begin() + h), y(s.begin() + h, s.end());
    return std::fabs(median_of(x) - median_of(y)) <= mad_of(s);
}

static const size_t CAP = 300;

int main()
{
    int failures = 0, capped = 0;
    const int steps = 300, every = 25;

    std::printf("== exactness, n=256, %d steps, injection every %d ==\n", steps, every);
    std::printf("areas beta  skip<0.5ulp==exact  skip<2ulp_differs  twostage==gather  desc_differs\n");
    int control_skip_seen = 0, control_desc_seen = 0;
    for (int areas : {1, 4})
        for (float beta : {0.3f, 1.0f, 3.0f}) {
            Params p{256, areas, 1.0f, 0.5f, beta, 0.3f, 1.0f, 10.0f};
            std::vector<float> K = ring_kernel(p.n, 7);
            Graph asc = build(K, p.n, false), desc = build(K, p.n, true);
            Device ref(asc, p, GATHER, -1.0f), half(asc, p, GATHER, 0.5f),
                two_ulp(asc, p, GATHER, 2.0f), two(asc, p, TWOSTAGE, -1.0f),
                dsc(desc, p, GATHER, -1.0f);
            for (Device *d : {&ref, &half, &two_ulp, &two, &dsc})
                drive_run(*d, steps, every, 11);
            bool e1 = same(ref, half), c1 = !same(ref, two_ulp);
            bool e2 = same(ref, two), c2 = !same(ref, dsc);
            control_skip_seen += c1;
            control_desc_seen += c2;
            failures += !e1 + !e2;
            std::printf("%-5d %-5.1f %-19s %-17s %-16s %s\n", areas, beta, e1 ? "yes" : "NO",
                        c1 ? "yes" : "no", e2 ? "yes" : "NO", c2 ? "yes" : "no");
        }
    if (!control_skip_seen) {
        std::printf("INVALID: the 2-ulp control never changed a trajectory\n");
        ++failures;
    }
    if (!control_desc_seen) {
        std::printf("INVALID: the descending control never changed a trajectory\n");
        ++failures;
    }

    std::printf("\n== quiet rows, beta=1.0, areas=1, over %d driven steps ==\n", steps);
    std::printf("n      terms_skipped (of all term visits)   quiet_rows_max_in_any_step (of n)\n");
    for (int n : {256, 1024, 4096}) {
        Params p{n, 1, 1.0f, 0.5f, 1.0f, 0.3f, 1.0f, 10.0f};
        std::vector<float> K = ring_kernel(n, 7);
        Graph g = build(K, n, false);
        Device d(g, p, GATHER, 0.5f);
        d.set_state(0.02f);
        uint32_t seed = 11;
        std::vector<double> quiet;
        unsigned long long total_skip = 0, total_terms = 0;
        std::vector<int> used(n);
        int quiet_max = 0;
        for (long s = 0; s < steps; ++s) {
            if (s % every == 0) {
                seed = seed * 1664525u + 1013904223u;
                d.inject((int)((seed >> 8) % (uint32_t)n), 0.2f + (seed & 0xFF) / 255.0f);
            }
            CHECK(hipMemset(d.skipped, 0, sizeof(unsigned long long)));
            CHECK(hipMemset(d.row_used, 0, n * sizeof(int)));
            d.step(true);
            unsigned long long sk = 0;
            CHECK(hipMemcpy(&sk, d.skipped, sizeof sk, hipMemcpyDeviceToHost));
            CHECK(hipMemcpy(used.data(), d.row_used, n * sizeof(int), hipMemcpyDeviceToHost));
            int q = 0;
            for (int x : used)
                q += x == 0;
            quiet_max = std::max(quiet_max, q);
            quiet.push_back(100.0 * q / n);
            total_skip += sk;
            total_terms += g.src.size();
        }
        std::printf("%-6d %llu of %llu                     %d of %d\n", n, total_skip,
                    total_terms, quiet_max, n);
    }

    std::printf("\n== per-step time, batches of back-to-back steps, from a settled state ==\n");
    std::printf("n      kind      ms/step    samples  stopped_on\n");
    hipEvent_t t0, t1;
    CHECK(hipEventCreate(&t0));
    CHECK(hipEventCreate(&t1));
    for (int n : {1024, 4096, 16384}) {
        Params p{n, 1, 1.0f, 0.5f, 1.0f, 0.3f, 1.0f, 10.0f};
        std::vector<float> K = ring_kernel(n, 7);
        Graph g = build(K, n, false);
        for (Kind k : {GATHER, TWOSTAGE}) {
            Device d(g, p, k, -1.0f);
            drive_run(d, steps, every, 11);
            const int batch = 100;
            std::vector<double> s;
            while (!stable(s) && s.size() < CAP) {
                CHECK(hipEventRecord(t0));
                for (int b = 0; b < batch; ++b)
                    d.step();
                CHECK(hipEventRecord(t1));
                CHECK(hipEventSynchronize(t1));
                float ms = 0.0f;
                CHECK(hipEventElapsedTime(&ms, t0, t1));
                s.push_back(ms / batch);
            }
            bool hit = !stable(s);
            capped += hit;
            std::printf("%-6d %-9s %9.5f  %7zu  %s\n", n, k == GATHER ? "gather" : "twostage",
                        median_of(s), s.size(), hit ? "cap" : "stable");
        }
    }
    CHECK(hipEventDestroy(t0));
    CHECK(hipEventDestroy(t1));

    if (failures) {
        std::printf("FAIL: %d check(s) failed\n", failures);
        return 1;
    }
    if (capped) {
        std::printf("RESULT: %d timing(s) ended on the cap -- not a measurement\n", capped);
        return 2;
    }
    std::printf("PASS: field_quiet\n");
    return 0;
}
