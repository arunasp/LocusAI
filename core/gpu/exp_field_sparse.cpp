// Field.step with a sparse kernel against the dense one.
//
// Sparse layout gathers by target: for each unit j, its incoming
// (source i, weight) pairs sorted by ascending i. Each thread sums its own
// list in the order field.py and the dense kernel use, with no atomics, so
// the two trajectories must be bit-identical (the dense kernel's extra
// terms are exact zeros). A descending-order variant is the negative
// control: it must differ, or the comparison cannot see summation order.
//
// Timing starts every step from one saved state, so the work per step is
// fixed and the number of active source rows is printed with it.
// Per-area totals use carried per-block partials (see exp_field_step.cpp).
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
static const float THRESHOLD = 0.05f;

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
    float net = (p.rho * aj + p.g * drive) / (1.0f + p.beta * (total - aj));
    float v = aj + p.dt * (net - p.leak * aj);
    return v < 0.0f ? 0.0f : (v > p.amax ? p.amax : v);
}

__global__ void dense_step(const float *__restrict__ K, const float *__restrict__ a,
                           float *__restrict__ nxt, const float *__restrict__ part_in,
                           float *__restrict__ part_out, Params p)
{
    __shared__ float sh[BD];
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    float drive = 0.0f;
    for (int base = 0; base < p.n; base += blockDim.x) {
        int i = base + threadIdx.x;
        sh[threadIdx.x] = i < p.n ? a[i] : 0.0f;
        __syncthreads();
        int lim = min((int)blockDim.x, p.n - base);
        if (j < p.n)
            for (int t = 0; t < lim; ++t) {
                float ai = sh[t];
                if (ai > FLOOR)
                    drive += ai * K[(size_t)(base + t) * p.n + j];
            }
        __syncthreads();
    }
    float v = 0.0f;
    if (j < p.n) {
        v = finish(a[j], drive, j, part_in, p);
        nxt[j] = v;
    }
    block_partials(v, j, p.n, p.areas, sh, part_out);
}

__global__ void sparse_step(const int *__restrict__ colptr, const int *__restrict__ src,
                            const float *__restrict__ w, const float *__restrict__ a,
                            float *__restrict__ nxt, const float *__restrict__ part_in,
                            float *__restrict__ part_out, Params p)
{
    __shared__ float sh[BD];
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    float v = 0.0f;
    if (j < p.n) {
        float drive = 0.0f;
        for (int k = colptr[j]; k < colptr[j + 1]; ++k) {
            float ai = a[src[k]];
            if (ai > FLOOR)
                drive += ai * w[k];
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

struct Csc {
    std::vector<int> colptr, src;
    std::vector<float> w;
};

static Csc to_csc(const std::vector<float> &K, int n, bool descending)
{
    Csc c;
    c.colptr.push_back(0);
    for (int j = 0; j < n; ++j) {
        std::vector<std::pair<int, float>> col;
        for (int i = 0; i < n; ++i) {
            float x = K[(size_t)i * n + j];
            if (x != 0.0f)
                col.push_back({i, x});
        }
        if (descending)
            std::reverse(col.begin(), col.end());
        for (auto &e : col) {
            c.src.push_back(e.first);
            c.w.push_back(e.second);
        }
        c.colptr.push_back((int)c.src.size());
    }
    return c;
}

enum Kind { DENSE, SPARSE };

struct Device {
    Params p;
    int blocks;
    Kind kind;
    float *K = nullptr, *w = nullptr, *a = nullptr, *b = nullptr, *saved = nullptr;
    int *colptr = nullptr, *src = nullptr;
    float *part[2] = {nullptr, nullptr};

    Device(Kind k, const std::vector<float> &Kh, const Csc *c, Params p_) : p(p_), kind(k)
    {
        blocks = (p.n + BD - 1) / BD;
        if (kind == DENSE) {
            size_t nn = (size_t)p.n * p.n;
            CHECK(hipMalloc(&K, nn * sizeof(float)));
            CHECK(hipMemcpy(K, Kh.data(), nn * sizeof(float), hipMemcpyHostToDevice));
        } else {
            CHECK(hipMalloc(&colptr, c->colptr.size() * sizeof(int)));
            CHECK(hipMalloc(&src, c->src.size() * sizeof(int)));
            CHECK(hipMalloc(&w, c->w.size() * sizeof(float)));
            CHECK(hipMemcpy(colptr, c->colptr.data(), c->colptr.size() * sizeof(int), hipMemcpyHostToDevice));
            CHECK(hipMemcpy(src, c->src.data(), c->src.size() * sizeof(int), hipMemcpyHostToDevice));
            CHECK(hipMemcpy(w, c->w.data(), c->w.size() * sizeof(float), hipMemcpyHostToDevice));
        }
        CHECK(hipMalloc(&a, p.n * sizeof(float)));
        CHECK(hipMalloc(&b, p.n * sizeof(float)));
        CHECK(hipMalloc(&saved, p.n * sizeof(float)));
        for (auto &q : part)
            CHECK(hipMalloc(&q, blocks * p.areas * sizeof(float)));
    }
    ~Device()
    {
        // Teardown: a failure here cannot change any result already printed.
        for (void *q : {(void *)K, (void *)w, (void *)a, (void *)b, (void *)saved,
                        (void *)colptr, (void *)src, (void *)part[0], (void *)part[1]})
            if (q)
                (void)hipFree(q);
    }

    // part[0] always holds the partials of the current state.
    void set_state(float level)
    {
        std::vector<float> h(p.n, level);
        CHECK(hipMemcpy(a, h.data(), p.n * sizeof(float), hipMemcpyHostToDevice));
        reduce_kernel<<<blocks, BD>>>(a, part[0], p);
        CHECK(hipGetLastError());
    }

    void step()
    {
        if (kind == DENSE)
            dense_step<<<blocks, BD>>>(K, a, b, part[0], part[1], p);
        else
            sparse_step<<<blocks, BD>>>(colptr, src, w, a, b, part[0], part[1], p);
        CHECK(hipGetLastError());
        std::swap(a, b);
        std::swap(part[0], part[1]);
    }

    void inject(int idx, float amt)
    {
        inject_kernel<<<blocks, BD>>>(a, part[0], idx, amt, p);
        CHECK(hipGetLastError());
    }

    void save() { CHECK(hipMemcpy(saved, a, p.n * sizeof(float), hipMemcpyDeviceToDevice)); }

    void restore()
    {
        CHECK(hipMemcpy(a, saved, p.n * sizeof(float), hipMemcpyDeviceToDevice));
        reduce_kernel<<<blocks, BD>>>(a, part[0], p);
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

static int active_rows(const std::vector<float> &a)
{
    int c = 0;
    for (float x : a)
        c += x > FLOOR;
    return c;
}

static std::vector<int> support(const std::vector<float> &a)
{
    std::vector<int> s;
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i] > THRESHOLD)
            s.push_back((int)i);
    return s;
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

    std::printf("== correctness, n=256, %d steps, injection every %d ==\n", steps, every);
    std::printf("areas beta  sparse==dense  descending_differs  support_same\n");
    for (int areas : {1, 4})
        for (float beta : {0.3f, 1.0f, 3.0f}) {
            Params p{256, areas, 1.0f, 0.5f, beta, 0.3f, 1.0f, 10.0f};
            std::vector<float> K = ring_kernel(p.n, 7);
            Csc asc = to_csc(K, p.n, false), desc = to_csc(K, p.n, true);
            Device dd(DENSE, K, nullptr, p), sd(SPARSE, K, &asc, p), cd(SPARSE, K, &desc, p);
            drive_run(dd, steps, every, 11);
            drive_run(sd, steps, every, 11);
            drive_run(cd, steps, every, 11);
            auto x = dd.state(), y = sd.state(), z = cd.state();
            bool ident = std::memcmp(x.data(), y.data(), p.n * sizeof(float)) == 0;
            bool diff = std::memcmp(x.data(), z.data(), p.n * sizeof(float)) != 0;
            bool sup = support(x) == support(y);
            std::printf("%-5d %-5.1f %-14s %-19s %s\n", areas, beta, ident ? "yes" : "NO",
                        diff ? "yes" : "NO", sup ? "yes" : "NO");
            failures += !ident + !sup;
            if (!diff)
                std::printf("  control not detected: equality above says nothing here\n");
            failures += !diff;
        }

    std::printf("\n== per-step time from a fixed state, areas=1, beta=1.0 ==\n");
    std::printf("n      state     active_rows  kind    ms/step    samples  stopped_on\n");
    hipEvent_t t0, t1;
    CHECK(hipEventCreate(&t0));
    CHECK(hipEventCreate(&t1));
    for (int n : {1024, 4096, 8192, 16384}) {
        Params p{n, 1, 1.0f, 0.5f, 1.0f, 0.3f, 1.0f, 10.0f};
        std::vector<float> K = ring_kernel(n, 7);
        Csc asc = to_csc(K, n, false);
        Device dd(DENSE, K, nullptr, p), sd(SPARSE, K, &asc, p);
        for (int st = 0; st < 2; ++st) {
            // state 0: every unit at 0.02; state 1: after 300 driven steps.
            for (Device *d : {&dd, &sd}) {
                if (st == 0)
                    d->set_state(0.02f);
                else
                    drive_run(*d, steps, every, 11);
                d->save();
            }
            int act = active_rows(dd.state());
            for (Device *d : {&dd, &sd}) {
                std::vector<double> s;
                while (!stable(s) && s.size() < CAP) {
                    d->restore();
                    CHECK(hipEventRecord(t0));
                    d->step();
                    CHECK(hipEventRecord(t1));
                    CHECK(hipEventSynchronize(t1));
                    float ms = 0.0f;
                    CHECK(hipEventElapsedTime(&ms, t0, t1));
                    s.push_back(ms);
                }
                bool hit = !stable(s);
                capped += hit;
                std::printf("%-6d %-9s %-12d %-7s %9.4f  %7zu  %s\n", n,
                            st == 0 ? "uniform" : "settled", act,
                            d->kind == DENSE ? "dense" : "sparse", median_of(s), s.size(),
                            hit ? "cap" : "stable");
            }
        }
    }
    CHECK(hipEventDestroy(t0));
    CHECK(hipEventDestroy(t1));

    if (failures) {
        std::printf("FAIL: %d correctness check(s) failed\n", failures);
        return 1;
    }
    if (capped) {
        std::printf("RESULT: %d timing(s) ended on the cap -- not a measurement\n", capped);
        return 2;
    }
    std::printf("PASS: field_sparse\n");
    return 0;
}
