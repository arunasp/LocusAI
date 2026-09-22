/* Host-side memory hierarchy bandwidth: CPU cache sweep, DRAM write, and
 * sequential disk I/O on the filesystem the probe runs from.
 *
 * Each figure is sampled until the median of the first half of the
 * samples agrees with the median of the second half within the spread of
 * all of them; a figure that never stabilises ends on the cap and says so.
 *
 * All CPU figures are single-threaded: what one thread gets, not the
 * aggregate across cores.
 *
 * Disk I/O uses O_DIRECT, which bypasses the Linux page cache only. Under
 * WSL2 the filesystem is an image file on Windows, whose own cache may
 * still serve reads; the probe cannot tell the two apart.
 */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define CAP 300
#define MINS 6

static double now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static int cmp(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static double median(const double *v, int n)
{
    double *c = malloc(n * sizeof(double));
    memcpy(c, v, n * sizeof(double));
    qsort(c, n, sizeof(double), cmp);
    double m = n % 2 ? c[n / 2] : 0.5 * (c[n / 2 - 1] + c[n / 2]);
    free(c);
    return m;
}

static double mad(const double *v, int n)
{
    double md = median(v, n), *d = malloc(n * sizeof(double));
    for (int i = 0; i < n; ++i)
        d[i] = fabs(v[i] - md);
    double r = median(d, n);
    free(d);
    return r;
}

static int stable(const double *s, int n)
{
    if (n < MINS)
        return 0;
    int h = n / 2;
    return fabs(median(s, h) - median(s + h, n - h)) <= mad(s, n);
}

typedef double (*sample_fn)(void *ctx);

static int capped_total;

/* Returns GB/s median; prints one line. */
static double measure(const char *label, sample_fn f, void *ctx)
{
    double s[CAP];
    int n = 0;
    while (n < CAP && !stable(s, n))
        s[n++] = f(ctx);
    int hit = !stable(s, n);
    capped_total += hit;
    double m = median(s, n);
    printf("%-26s %9.1f GB/s  n=%-3d mad=%.1f  stopped_on=%s\n", label, m, n,
           mad(s, n), hit ? "cap" : "stable");
    fflush(stdout);
    return m;
}

/* ---- reads over a working set -------------------------------------- */

struct rd {
    const uint64_t *buf;
    size_t words;
    size_t passes;
};

static volatile uint64_t sink;

static double read_sample(void *p)
{
    struct rd *r = p;
    uint64_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    double t = now();
    for (size_t k = 0; k < r->passes; ++k)
        for (size_t i = 0; i + 3 < r->words; i += 4) {
            a0 += r->buf[i];
            a1 += r->buf[i + 1];
            a2 += r->buf[i + 2];
            a3 += r->buf[i + 3];
        }
    t = now() - t;
    sink = a0 + a1 + a2 + a3;
    return (double)r->words * 8 * r->passes / t / 1e9;
}

struct wr {
    char *buf;
    size_t bytes;
};

static double write_sample(void *p)
{
    struct wr *w = p;
    static int v;
    double t = now();
    memset(w->buf, ++v, w->bytes);
    t = now() - t;
    sink = (uint64_t)w->buf[w->bytes - 1];
    return (double)w->bytes / t / 1e9;
}

/* ---- disk ---------------------------------------------------------- */

struct dk {
    const char *path;
    char *buf;
    size_t block, blocks;
    int write;
};

static double disk_sample(void *p)
{
    struct dk *d = p;
    int fd = open(d->path, (d->write ? O_WRONLY | O_CREAT | O_TRUNC : O_RDONLY) | O_DIRECT, 0600);
    if (fd < 0) {
        fprintf(stderr, "FAIL: open %s: %s\n", d->path, strerror(errno));
        exit(1);
    }
    double t = now();
    for (size_t b = 0; b < d->blocks; ++b) {
        ssize_t r = d->write ? write(fd, d->buf, d->block) : read(fd, d->buf, d->block);
        if (r != (ssize_t)d->block) {
            fprintf(stderr, "FAIL: %s: %s\n", d->write ? "write" : "read", strerror(errno));
            exit(1);
        }
    }
    if (d->write && fsync(fd) != 0) {
        fprintf(stderr, "FAIL: fsync: %s\n", strerror(errno));
        exit(1);
    }
    t = now() - t;
    close(fd);
    return (double)d->block * d->blocks / t / 1e9;
}

static void print_caches(void)
{
    for (int i = 0; i < 8; ++i) {
        char path[128], lvl[16] = "", typ[32] = "", sz[32] = "";
        FILE *f;
        snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu0/cache/index%d/level", i);
        if (!(f = fopen(path, "r")))
            break;
        if (!fgets(lvl, sizeof lvl, f)) lvl[0] = 0;
        fclose(f);
        snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu0/cache/index%d/type", i);
        if ((f = fopen(path, "r"))) { if (!fgets(typ, sizeof typ, f)) typ[0] = 0; fclose(f); }
        snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu0/cache/index%d/size", i);
        if ((f = fopen(path, "r"))) { if (!fgets(sz, sizeof sz, f)) sz[0] = 0; fclose(f); }
        lvl[strcspn(lvl, "\n")] = typ[strcspn(typ, "\n")] = sz[strcspn(sz, "\n")] = 0;
        printf("cache L%s %-12s %s\n", lvl, typ, sz);
    }
}

int main(int argc, char **argv)
{
    const char *dir = argc > 1 ? argv[1] : ".";
    print_caches();

    /* Read sweep: each sample covers at least 256 MiB of reads so a
     * sample is long enough to time, whatever the working set. */
    const size_t max_bytes = (size_t)1 << 30;
    uint64_t *buf = aligned_alloc(4096, max_bytes);
    if (!buf) {
        fprintf(stderr, "FAIL: allocation\n");
        return 1;
    }
    for (size_t i = 0; i < max_bytes / 8; ++i)
        buf[i] = i;
    printf("\n== CPU read sweep (one thread) ==\n");
    for (size_t bytes = 16 << 10; bytes <= max_bytes; bytes <<= 1) {
        struct rd r = {buf, bytes / 8, ((size_t)256 << 20) / bytes};
        if (r.passes == 0)
            r.passes = 1;
        char label[64];
        if (bytes < (1 << 20))
            snprintf(label, sizeof label, "read %zu KiB", bytes >> 10);
        else
            snprintf(label, sizeof label, "read %zu MiB", bytes >> 20);
        measure(label, read_sample, &r);
    }
    printf("\n== CPU write (one thread) ==\n");
    struct wr w = {(char *)buf, max_bytes};
    measure("write 1024 MiB (memset)", write_sample, &w);

    printf("\n== disk, O_DIRECT, %s ==\n", dir);
    char path[4096];
    snprintf(path, sizeof path, "%s/.exp_hierarchy.tmp", dir);
    struct dk d = {path, (char *)buf, (size_t)4 << 20, 64, 1}; /* 256 MiB */
    measure("disk write 256 MiB + fsync", disk_sample, &d);
    d.write = 0;
    measure("disk read 256 MiB", disk_sample, &d);
    if (unlink(path) != 0)
        fprintf(stderr, "WARN: could not remove %s: %s\n", path, strerror(errno));
    printf("[removed] %s\n", path);
    free(buf);

    if (capped_total) {
        printf("RESULT: %d figure(s) ended on the cap -- not a measurement\n", capped_total);
        return 2;
    }
    printf("PASS: hierarchy\n");
    return 0;
}
