// Field.step on the device: correctness, single-launch carried inhibition
// sums, and FP8 kernel weights.
//
// Mirrors core/py/locus/field.py (noise = 0): every unit reads the state at
// entry; drive[j] = sum_i a[i] * K[i][j] in ascending i; shunting inhibition
// divides by 1 + beta * (area total - a[j]); leak; clamp to [0, a_max].
//
// Per-area totals are per-block partial sums over a fixed-order tree.
//   twolaunch  reduce kernel over the entry state, then the update kernel.
//   carried    the update kernel writes the partials of what it just wrote;
//              the next step reads them. One launch per step.
//   lagged     negative control: reads partials one step older.
// carried must be bit-identical to twolaunch (same values, same tree, same
// order); lagged must not be, or the comparison cannot see a lag.
//
// FP8 E4M3 weights are decoded through a 256-entry table.
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

__constant__ float c_lut[256];

__device__ inline float wload(const float *K, size_t i) { return K[i]; }
__device__ inline float wload(const uint8_t *K, size_t i) { return c_lut[K[i]]; }

// Fixed-order tree over the block, one partial per area. Areas interleave:
// unit j belongs to area j % areas, as in the CPU experiments.
__device__ void block_partials(float v, int j, int n, int areas,
                               float *sh, float *part)
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

template <typename W, bool WRITE_PART>
__global__ void step_kernel(const W *__restrict__ K, const float *__restrict__ a,
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
                    drive += ai * wload(K, (size_t)(base + t) * p.n + j);
            }
        __syncthreads();
    }
    float v = 0.0f;
    if (j < p.n) {
        int area = j % p.areas;
        float total = 0.0f;
        for (int b = 0; b < (int)gridDim.x; ++b)
            total += part_in[b * p.areas + area];
        float aj = a[j];
        float net = (p.rho * aj + p.g * drive) / (1.0f + p.beta * (total - aj));
        v = aj + p.dt * (net - p.leak * aj);
        v = v < 0.0f ? 0.0f : (v > p.amax ? p.amax : v);
        nxt[j] = v;
    }
    if (WRITE_PART)
        block_partials(v, j, p.n, p.areas, sh, part_out);
}

// Adds to one unit and recomputes its block's partials with the same tree,
// so an injection leaves carried partials exactly as a fresh reduce would.
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

static float e4m3_value(uint8_t b)
{
    int e = (b >> 3) & 0xF, m = b & 7;
    if (b & 0x80)
        return NAN;                       // weights are non-negative
    if (e == 0xF && m == 7)
        return NAN;
    if (e == 0)
        return std::ldexp((float)m / 8.0f, -6);
    return std::ldexp(1.0f + (float)m / 8.0f, e - 7);
}

static std::vector<float> make_lut()
{
    std::vector<float> lut(256);
    for (int b = 0; b < 256; ++b)
        lut[b] = e4m3_value((uint8_t)b);
    return lut;
}

static uint8_t e4m3_encode(float x, const std::vector<float> &lut)
{
    uint8_t best = 0;
    float bd = INFINITY;
    for (int b = 0; b < 127; ++b) {       // finite non-negative codes
        float d = std::fabs(lut[b] - x);
        if (d < bd) {
            bd = d;
            best = (uint8_t)b;
        }
    }
    return best;
}

// Ring kernel as in core/tests/test_cycle.py: self term plus decaying
// neighbours out to `reach`, rows normalised. Own LCG, so the values
// differ from the Python ones; the CPU replica below uses the same matrix.
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
            float w = std::pow(0.6f, (float)d) * (0.4f + rnd());
            row[(i + d) % n] += w;
            row[(i - d + n) % n] += w;
        }
        float s = 0.0f;
        for (int k = 0; k < n; ++k)
            s += row[k];
        for (int k = 0; k < n; ++k)
            row[k] /= s;
    }
    return K;
}

// Double-precision replica of Field.step for the correctness check.
static void cpu_step(const std::vector<float> &K, std::vector<double> &a, const Params &p)
{
    int n = p.n;
    std::vector<double> drive(n, 0.0), tot(p.areas, 0.0), nxt(n);
    for (int i = 0; i < n; ++i)
        if (a[i] > FLOOR)
            for (int j = 0; j < n; ++j)
                drive[j] += a[i] * K[(size_t)i * n + j];
    for (int i = 0; i < n; ++i)
        tot[i % p.areas] += a[i];
    for (int j = 0; j < n; ++j) {
        double aj = a[j];
        double net = (p.rho * aj + p.g * drive[j]) / (1.0 + p.beta * (tot[j % p.areas] - aj));
        double v = aj + p.dt * (net - p.leak * aj);
        nxt[j] = v < 0.0 ? 0.0 : (v > p.amax ? p.amax : v);
    }
    a = nxt;
}

enum Mode { TWOLAUNCH, CARRIED, LAGGED };

struct Device {
    Params p;
    int blocks;
    void *K = nullptr;
    bool fp8 = false;
    float *a = nullptr, *b = nullptr;
    float *part[3] = {nullptr, nullptr, nullptr};

    Device(const std::vector<float> &Kh, Params p_, bool fp8_,
           const std::vector<float> &lut)
        : p(p_), fp8(fp8_)
    {
        blocks = (p.n + BD - 1) / BD;
        size_t nn = (size_t)p.n * p.n;
        if (fp8) {
            std::vector<uint8_t> q(nn);
            for (size_t i = 0; i < nn; ++i)
                q[i] = e4m3_encode(Kh[i], lut);
            CHECK(hipMalloc(&K, nn));
            CHECK(hipMemcpy(K, q.data(), nn, hipMemcpyHostToDevice));
        } else {
            CHECK(hipMalloc(&K, nn * sizeof(float)));
            CHECK(hipMemcpy(K, Kh.data(), nn * sizeof(float), hipMemcpyHostToDevice));
        }
        CHECK(hipMalloc(&a, p.n * sizeof(float)));
        CHECK(hipMalloc(&b, p.n * sizeof(float)));
        for (auto &q : part)
            CHECK(hipMalloc(&q, blocks * p.areas * sizeof(float)));
    }
    ~Device()
    {
        // Teardown: a failure here cannot change any result already printed.
        (void)hipFree(K);
        (void)hipFree(a);
        (void)hipFree(b);
        for (auto q : part)
            (void)hipFree(q);
    }

    void set_state(float level)
    {
        std::vector<float> h(p.n, level);
        CHECK(hipMemcpy(a, h.data(), p.n * sizeof(float), hipMemcpyHostToDevice));
        reduce_kernel<<<blocks, BD>>>(a, part[0], p);
        CHECK(hipGetLastError());
        reduce_kernel<<<blocks, BD>>>(a, part[1], p);
        reduce_kernel<<<blocks, BD>>>(a, part[2], p);
        CHECK(hipGetLastError());
    }

    template <typename W>
    void launch(Mode m, long s)
    {
        const W *Kw = (const W *)K;
        if (m == TWOLAUNCH) {
            reduce_kernel<<<blocks, BD>>>(a, part[0], p);
            step_kernel<W, false><<<blocks, BD>>>(Kw, a, b, part[0], nullptr, p);
        } else {
            // Buffer s % 3 holds the partials of the current state;
            // (s + 2) % 3 holds those of the state one step older.
            const float *in = part[m == CARRIED ? s % 3 : (s + 2) % 3];
            step_kernel<W, true><<<blocks, BD>>>(Kw, a, b, in, part[(s + 1) % 3], p);
        }
        CHECK(hipGetLastError());
        std::swap(a, b);
    }

    void step(Mode m, long s)
    {
        if (fp8)
            launch<uint8_t>(m, s);
        else
            launch<float>(m, s);
    }

    void inject(Mode m, long s, int idx, float amt)
    {
        // The current state's partials live in part[s % 3] for the carried
        // modes; twolaunch recomputes them itself every step.
        inject_kernel<<<blocks, BD>>>(a, part[m == TWOLAUNCH ? 0 : s % 3], idx, amt, p);
        CHECK(hipGetLastError());
    }

    std::vector<float> state()
    {
        std::vector<float> h(p.n);
        CHECK(hipMemcpy(h.data(), a, p.n * sizeof(float), hipMemcpyDeviceToHost));
        return h;
    }
};

static std::vector<float> run(Device &d, Mode m, int steps, int every, uint32_t seed)
{
    d.set_state(0.02f);
    for (long s = 0; s < steps; ++s) {
        if (s % every == 0) {
            seed = seed * 1664525u + 1013904223u;
            int idx = (int)((seed >> 8) % (uint32_t)d.p.n);
            d.inject(m, s, idx, 0.2f + (seed & 0xFF) / 255.0f);
        }
        d.step(m, s);
    }
    CHECK(hipDeviceSynchronize());
    return d.state();
}

static std::vector<double> run_cpu(const std::vector<float> &K, const Params &p,
                                   int steps, int every, uint32_t seed)
{
    std::vector<double> a(p.n, 0.02);
    for (long s = 0; s < steps; ++s) {
        if (s % every == 0) {
            seed = seed * 1664525u + 1013904223u;
            int idx = (int)((seed >> 8) % (uint32_t)p.n);
            a[idx] = std::min((double)p.amax, a[idx] + (0.2f + (seed & 0xFF) / 255.0f));
        }
        cpu_step(K, a, p);
    }
    return a;
}

template <class V>
static std::vector<int> support(const V &a)
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

static const size_t CAP = 200;

int main()
{
    std::vector<float> lut = make_lut();
    CHECK(hipMemcpyToSymbol(HIP_SYMBOL(c_lut), lut.data(), 256 * sizeof(float)));

    int failures = 0, capped = 0;
    const int steps = 300, every = 25;

    std::printf("== correctness, n=256, %d steps, injection every %d ==\n", steps, every);
    std::printf("areas beta  carried==twolaunch  lagged_differs  fp32_vs_cpu(maxdiff,support)  fp8_vs_fp32_support\n");
    for (int areas : {1, 4}) {
        for (float beta : {0.3f, 1.0f, 3.0f}) {
            Params p{256, areas, 1.0f, 0.5f, beta, 0.3f, 1.0f, 10.0f};
            std::vector<float> K = ring_kernel(p.n, 7);
            Device f32(K, p, false, lut), f8(K, p, true, lut);
            auto two = run(f32, TWOLAUNCH, steps, every, 11);
            auto car = run(f32, CARRIED, steps, every, 11);
            auto lag = run(f32, LAGGED, steps, every, 11);
            auto q8 = run(f8, CARRIED, steps, every, 11);
            auto ref = run_cpu(K, p, steps, every, 11);
            bool ident = std::memcmp(two.data(), car.data(), p.n * sizeof(float)) == 0;
            bool lag_diff = std::memcmp(two.data(), lag.data(), p.n * sizeof(float)) != 0;
            double md = 0.0;
            for (int i = 0; i < p.n; ++i)
                md = std::max(md, std::fabs((double)car[i] - ref[i]));
            bool sup = support(car) == support(ref);
            bool sup8 = support(q8) == support(car);
            std::printf("%-5d %-5.1f %-19s %-15s %.2e %-19s %s\n", areas, beta,
                        ident ? "yes" : "NO", lag_diff ? "yes" : "NO", md,
                        sup ? "same" : "DIFFERENT", sup8 ? "same" : "different");
            failures += !ident + !lag_diff + !sup;
        }
    }

    std::printf("\n== per-step time, areas=1, beta=1.0 ==\n");
    std::printf("n      variant          ms/step   K GB/s   samples  stopped_on\n");
    hipEvent_t t0, t1;
    CHECK(hipEventCreate(&t0));
    CHECK(hipEventCreate(&t1));
    for (int n : {1024, 4096, 8192}) {
        Params p{n, 1, 1.0f, 0.5f, 1.0f, 0.3f, 1.0f, 10.0f};
        std::vector<float> K = ring_kernel(n, 7);
        struct V { const char *name; bool fp8; Mode m; };
        for (V v : {V{"fp32 twolaunch", false, TWOLAUNCH}, V{"fp32 carried", false, CARRIED},
                    V{"fp8  carried", true, CARRIED}}) {
            Device d(K, p, v.fp8, lut);
            d.set_state(0.02f);
            d.inject(v.m, 0, 0, 1.0f);
            const int chunk = 20;
            long s = 0;
            for (int w = 0; w < chunk; ++w)
                d.step(v.m, s++);
            std::vector<double> samples;
            while (!stable(samples) && samples.size() < CAP) {
                CHECK(hipEventRecord(t0));
                for (int w = 0; w < chunk; ++w)
                    d.step(v.m, s++);
                CHECK(hipEventRecord(t1));
                CHECK(hipEventSynchronize(t1));
                float ms = 0.0f;
                CHECK(hipEventElapsedTime(&ms, t0, t1));
                samples.push_back(ms / chunk);
            }
            bool hit = !stable(samples);
            capped += hit;
            double ms = median_of(samples);
            double kbytes = (double)n * n * (v.fp8 ? 1.0 : 4.0);
            std::printf("%-6d %-16s %8.4f  %7.1f  %7zu  %s\n", n, v.name, ms,
                        kbytes / (ms * 1.0e6), samples.size(), hit ? "cap" : "stable");
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
    std::printf("PASS: field_step\n");
    return 0;
}
