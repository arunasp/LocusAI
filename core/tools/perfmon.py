"""Sample host CPU and memory use while a job runs; summarise afterwards.

  perfmon.py record CSV [INTERVAL_S]   sample until SIGTERM/SIGINT
  perfmon.py summary CSV [DEVICE_S]    print a utilisation summary

Samples /proc/stat per CPU (busy = everything but idle and iowait) and
/proc/meminfo.

GPU COUNTERS ARE RESOLVED AT INVOCATION, not assumed. Which management
interface answers depends on what the environment EXPOSES, not on which
library is installed: `/dev/kfd` plus `/dev/dri` is the native path and
amd-smi talks to the amdgpu driver; `/dev/dxg` is WSL2, where that
driver does not exist and the dxg build of libamd_smi is the one that
answers. Both halves have to be present AND matched to the exposed
device -- this file used to state flatly that GPU counters are
unreadable under WSL2, which bakes one boot's answer into a tool that
runs on both.

When no interface answers, device use still gets reported: the job
passes its own measured DEVICE_S, device seconds, and the summary gives
it as a share of wall time. That number is better than a utilisation
percentage anyway -- it says how much of the run was serial host work.
"""

import csv
import os
import signal
import subprocess
import sys
import time


def gpu_sampler():
    """A callable returning (busy_pct, mem_mib), or None if nothing here
    can answer. Probed once, because the answer cannot change mid-run.

    Returns None quietly: a job on a machine with no management
    interface should still produce a perf file, and the summary falls
    back to the device seconds the job measured itself.
    """
    exposed = [d for d in ("/dev/kfd", "/dev/dri", "/dev/dxg")
               if os.path.exists(d)]
    if not exposed:
        return None
    smi = None
    for cand in ("amd-smi", "rocm-smi"):
        for root in (os.environ.get("ROCM_PATH", "/opt/rocm"), ""):
            path = os.path.join(root, "bin", cand) if root else cand
            try:
                r = subprocess.run([path, "version"], capture_output=True,
                                   text=True, timeout=20)
            except (OSError, subprocess.SubprocessError):
                continue
            if r.returncode == 0:
                smi = path
                break
        if smi:
            break
    if not smi:
        return None

    def sample():
        try:
            r = subprocess.run([smi, "metric", "--usage", "--csv"],
                               capture_output=True, text=True, timeout=20)
            if r.returncode != 0:
                return None
            rows = list(csv.DictReader(r.stdout.splitlines()))
            if not rows:
                return None
            row = rows[0]
            busy = next((float(v) for k, v in row.items()
                         if "gfx_activity" in k), None)
            mem = next((float(v) for k, v in row.items()
                        if "used_vram" in k), None)
            return (busy, mem)
        except (OSError, ValueError, subprocess.SubprocessError):
            return None

    return sample if sample() else None


def cpu_times():
    """{cpu: (busy, total)} jiffies from /proc/stat."""
    out = {}
    with open("/proc/stat") as fh:
        for line in fh:
            if not line.startswith("cpu") or line.startswith("cpu "):
                continue
            f = line.split()
            v = [int(x) for x in f[1:]]
            idle = v[3] + (v[4] if len(v) > 4 else 0)
            out[f[0]] = (sum(v) - idle, sum(v))
    return out


def mem_used_mib():
    info = {}
    with open("/proc/meminfo") as fh:
        for line in fh:
            k, v = line.split(":", 1)
            info[k] = int(v.split()[0])
    return (info["MemTotal"] - info["MemAvailable"]) / 1024.0


def record(path, interval):
    stop = []
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.append(1))
    gpu = gpu_sampler()
    sys.stderr.write("perfmon: gpu counters %s\n"
                     % ("available" if gpu else
                        "unavailable here -- device seconds only"))
    prev, t_prev = cpu_times(), time.time()
    t0 = t_prev
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t", "busy_cores", "ncpu", "mem_mib",
                    "gpu_busy", "gpu_mib"])
        while not stop:
            time.sleep(interval)
            cur, t = cpu_times(), time.time()
            busy = 0.0
            for c, (b, tot) in cur.items():
                pb, pt = prev.get(c, (b, tot))
                if tot > pt:
                    busy += (b - pb) / (tot - pt)
            g = gpu() if gpu else None
            w.writerow(["%.2f" % (t - t0), "%.3f" % busy, len(cur),
                        "%.0f" % mem_used_mib(),
                        "" if not g or g[0] is None else "%.1f" % g[0],
                        "" if not g or g[1] is None else "%.0f" % g[1]])
            fh.flush()
            prev, t_prev = cur, t
    return 0


def summary(path, device_s=None):
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        print("perfmon: no samples in %s" % path)
        return 1
    busy = sorted(float(r["busy_cores"]) for r in rows)
    ncpu = int(rows[-1]["ncpu"])
    wall = float(rows[-1]["t"])
    n = len(busy)

    def pct(q):
        return busy[min(n - 1, int(q * (n - 1) + 0.5))]

    mean = sum(busy) / n
    low = sum(1 for b in busy if b < 0.25 * ncpu) / n
    print("perfmon: %d samples over %.0f s, %d CPUs" % (n, wall, ncpu))
    print("perfmon: busy cores mean %.1f (%.0f%%), p50 %.1f, p90 %.1f, "
          "max %.1f; below 25%% of CPUs for %.0f%% of the time"
          % (mean, 100 * mean / ncpu, pct(0.5), pct(0.9), busy[-1],
             100 * low))
    print("perfmon: CPU core-seconds %.0f of %.0f available"
          % (mean * wall, ncpu * wall))
    print("perfmon: peak memory %.0f MiB"
          % max(float(r["mem_mib"]) for r in rows))
    if device_s is not None and wall > 0:
        print("perfmon: GPU device %.1f s = %.1f%% of wall"
              % (device_s, 100 * device_s / wall))
    return 0


def main(argv):
    if len(argv) >= 2 and argv[0] == "record":
        return record(argv[1], float(argv[2]) if len(argv) > 2 else 1.0)
    if len(argv) >= 2 and argv[0] == "summary":
        return summary(argv[1], float(argv[2]) if len(argv) > 2 else None)
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
