"""Run the field experiments concurrently, one process each.

  run_exps.py OUTDIR [NAME ...]

Each experiment is an independent script whose loops are single-core by
nature (Field.step is O(n^2) in pure Python). Rather than edit scripts
whose printed output is a measurement record, this runs them side by
side: measured 2026-09-22, the six together take 111 s one after another
on one core of 24.

Each script's output goes to OUTDIR/<name>.log; the table reports wall
and CPU per script and the total. A non-zero exit from any script makes
this exit non-zero too.
"""

import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.join(HERE, "..", "tests")
sys.path.insert(0, TESTS)

from parallel import run_jobs  # noqa: E402

DEFAULT = ["exp_capacity", "exp_persistence", "exp_stack", "exp_singlepass",
           "exp_precision", "exp_stream", "exp_operating_point"]


def run(spec):
    name, outdir = spec
    path = os.path.join(TESTS, name + ".py")
    t0 = time.time()
    r = subprocess.run([sys.executable, path], capture_output=True,
                       text=True)
    wall = time.time() - t0
    log = os.path.join(outdir, name + ".log")
    with open(log, "w") as fh:
        fh.write(r.stdout)
        if r.stderr:
            fh.write("\n--- stderr ---\n" + r.stderr)
    return {"name": name, "rc": r.returncode, "wall": wall,
            "bytes": len(r.stdout), "log": log}


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    outdir, names = argv[0], argv[1:] or DEFAULT
    os.makedirs(outdir, exist_ok=True)
    missing = [n for n in names
               if not os.path.exists(os.path.join(TESTS, n + ".py"))]
    if missing:
        print("no such experiment: %s" % ", ".join(missing))
        return 2
    t0, c0 = time.time(), os.times()
    res = run_jobs(run, [(n, outdir) for n in names])
    c1, wall = os.times(), time.time() - t0
    cpu = ((c1.children_user + c1.children_system)
           - (c0.children_user + c0.children_system))
    print("%-22s %4s %9s  %s" % ("experiment", "rc", "wall", "log"))
    for r in res:
        print("%-22s %4d %8.1fs  %s%s" % (
            r["name"], r["rc"], r["wall"], r["log"],
            "   FAILED" if r["rc"] else ""))
    print("total wall %.1f s, child CPU %.1f s on %d processes "
          "(serial would be the CPU figure)" % (wall, cpu, len(names)))
    return 1 if any(r["rc"] for r in res) else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
