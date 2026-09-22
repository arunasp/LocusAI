"""Time the real tasks and keep every run, so gains are evidence.

  bench.py OUTDIR [NAME ...]        (default: every task below)

The execution layer (core/py/locus/execution.py) is work in progress
until its gain is proven over several tasks. One fast run proves
nothing: a figure moves with machine load, with what else is running in
the container, and with the corpus. So every run appends a row to
OUTDIR/history.csv -- commit, task, wall, CPU core-seconds, the task's
own reported figure -- and the table printed here compares this run with
the previous rows FOR THE SAME COMMIT-INDEPENDENT TASK, showing the
median of earlier runs rather than the last one.

A task is a shell command plus a regex that pulls its own headline
figure out of its output (bits per byte, seconds, whatever it prints),
so a speedup that changed a result is visible as a changed figure rather
than hidden behind a faster time.
"""

import csv
import os
import re
import subprocess
import sys
import time

TASKS = {
    "score-english": (
        "make prompt KNOW=build/know/english PROMPTARGS=--score",
        r"reader\s+([0-9.]+) bits per byte"),
    "experiments": (
        "make experiments",
        r"total wall ([0-9.]+) s"),
    "know-small": (
        "make know CORPUS=build/corpus-480e191 KNOW=build/know/bench-small",
        r"test ([0-9.]+) bits per byte"),
    "learn-gpu": (
        "make stream-learn-gpu CORPUS=build/corpus-480e191 "
        "GPUARGS='--only metaplastic'",
        r"cls, neocortex also metaplastic \(1/n\)\s+([0-9.-]+)"),
}


def commit():
    r = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                       capture_output=True, text=True)
    return r.stdout.strip() or "unknown"


def run(name, outdir):
    cmd, pattern = TASKS[name]
    t0, c0 = time.time(), os.times()
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    c1 = os.times()
    wall = time.time() - t0
    cpu = ((c1.children_user + c1.children_system)
           - (c0.children_user + c0.children_system))
    m = re.search(pattern, r.stdout)
    with open(os.path.join(outdir, name + ".log"), "w") as fh:
        fh.write(r.stdout + ("\n--- stderr ---\n" + r.stderr
                             if r.stderr else ""))
    return {"task": name, "rc": r.returncode, "wall": round(wall, 1),
            "cpu": round(cpu, 1), "figure": m.group(1) if m else ""}


def history(path):
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return list(csv.DictReader(fh))


def median(values):
    v = sorted(values)
    return v[len(v) // 2] if v else None


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    outdir, names = argv[0], argv[1:] or list(TASKS)
    unknown = [n for n in names if n not in TASKS]
    if unknown:
        print("no such task: %s (have: %s)"
              % (", ".join(unknown), ", ".join(sorted(TASKS))))
        return 2
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "history.csv")
    past = history(path)
    rows, head = [], commit()
    for name in names:
        rows.append(run(name, outdir))
    new = os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, ["when", "commit", "task", "rc", "wall",
                                "cpu", "figure"])
        if not new:
            w.writeheader()
        for r in rows:
            w.writerow(dict(r, when=time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                  time.gmtime()),
                            commit=head))
    print("%-14s %3s %8s %9s  %-16s %s"
          % ("task", "rc", "wall", "CPU", "figure", "vs median of past"))
    for r in rows:
        prev = [float(p["wall"]) for p in past
                if p["task"] == r["task"] and p["rc"] == "0"]
        med = median(prev)
        cmp = ("%+.0f%% of %d runs"
               % ((r["wall"] / med - 1.0) * 100.0, len(prev))
               if med else "first run")
        same = [p["figure"] for p in past if p["task"] == r["task"]
                and p["figure"]]
        if same and r["figure"] and r["figure"] != same[-1]:
            cmp += "   FIGURE CHANGED from %s" % same[-1]
        print("%-14s %3d %7.1fs %8.1fs  %-16s %s"
              % (r["task"], r["rc"], r["wall"], r["cpu"],
                 r["figure"] or "-", cmp))
    print("appended %d rows to %s" % (len(rows), path))
    return 1 if any(r["rc"] for r in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
