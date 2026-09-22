"""Which outcome signal should decide what reading keeps?

  exp_modulators.py KNOW CORPUS OUTDIR [--salt S]

KNOW is a generation-0 store trained on CORPUS's train split. Each method
gets its own copy under OUTDIR/<method>/, then reads the validation files
one at a time with learning on (one episode each); after every episode it
sets an outcome and sleeps, so the episode's tags are captured or lapse.
The test files are halved: the probe half supplies the external outcome,
the final half is scored by no method and is the only reported score.

Methods: none, all, novelty (episode surprise above its running mean),
familiarity (the opposite), progress (surprise falling within the
episode, above its running mean), external (probe improved), mixture
(weighted novelty and progress; the weights learn from the sign of the
external outcome after each decision at a fixed rate 0.5, starting at 0,
ties captured), and adaptive (the same cues, weights by recursive least
squares against the external outcome standardised by its own running
statistics: the step is the current uncertainty about the weights, large
while evidence is scarce and shrinking as it accumulates -- Dayan,
Kakade & Montague 2000). Both mixtures decide before they see the
external outcome. One process per method.
"""

import multiprocessing
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "py"))
sys.path.insert(0, HERE)

import exp_learn_stream as X  # noqa: E402
from locus.encode import repo_files  # noqa: E402
from locus.knowledge import Knowledge  # noqa: E402

METHODS = ["none", "all", "novelty", "familiarity", "progress",
           "external", "mixture", "adaptive"]
RATE = 0.5


def bpb(k, files):
    bits = n = 0
    for d in files:
        b = k.read(d, learn=False)
        bits += sum(b)
        n += len(b)
    return bits / max(n, 1)


class RLS:
    """Recursive least squares for outcome ~ w . cues. No learning rate:
    the gain is P z / (1 + z' P z), where P, the weight uncertainty,
    starts uninformative and shrinks with each observation."""

    def __init__(self, n, prior=1e6):
        self.w = [0.0] * n
        self.P = [[prior if i == j else 0.0 for j in range(n)]
                  for i in range(n)]

    def predict(self, z):
        return sum(wi * zi for wi, zi in zip(self.w, z))

    def update(self, z, y):
        n = len(z)
        Pz = [sum(self.P[i][j] * z[j] for j in range(n)) for i in range(n)]
        denom = 1.0 + sum(z[i] * Pz[i] for i in range(n))
        k = [v / denom for v in Pz]
        err = y - self.predict(z)
        self.w = [self.w[i] + k[i] * err for i in range(n)]
        self.P = [[self.P[i][j] - k[i] * Pz[j] for j in range(n)]
                  for i in range(n)]


class Running:
    """Running mean and standard deviation of past values."""

    def __init__(self):
        self.n, self.m, self.s = 0, 0.0, 0.0

    def z(self, x):
        if self.n < 2:
            return 0.0
        sd = (self.s / (self.n - 1)) ** 0.5
        return (x - self.m) / sd if sd > 0 else 0.0

    def add(self, x):
        self.n += 1
        d = x - self.m
        self.m += d / self.n
        self.s += d * (x - self.m)


def run(spec):
    method, know, outdir, episodes, probe, final = spec
    t0, c0 = time.time(), time.process_time()
    d = os.path.join(outdir, method)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "store")
    for ext in ("", ".json"):
        shutil.copyfile(know + ext, path + ext)
    for ext in (".tags", ".prev", ".prev.json"):
        if os.path.exists(path + ext):
            os.remove(path + ext)
    k = Knowledge(path)
    nov, prog = Running(), Running()
    w = [0.0, 0.0]
    rls, rew = RLS(2), Running()
    need_probe = method in ("external", "mixture", "adaptive")
    before = bpb(k, probe) if need_probe else None
    log, kept = [], 0
    for i, ep in enumerate(episodes):
        bits = k.read(ep)
        h = len(bits) // 2
        s_nov = sum(bits) / len(bits)
        s_prog = (sum(bits[:h]) / max(h, 1)
                  - sum(bits[h:]) / max(len(bits) - h, 1))
        z = [nov.z(s_nov), prog.z(s_prog)]
        after = bpb(k, probe) if need_probe else None
        r = (before - after) if need_probe else None
        if method == "none":
            o = 0.0
        elif method == "all":
            o = 1.0
        elif method == "novelty":
            o = 1.0 if z[0] > 0 or nov.n < 2 else -1.0
        elif method == "familiarity":
            o = 1.0 if z[0] < 0 or nov.n < 2 else -1.0
        elif method == "progress":
            o = 1.0 if z[1] > 0 or prog.n < 2 else -1.0
        elif method == "external":
            o = 1.0 if r > 0 else -1.0
        elif method == "mixture":
            m = w[0] * z[0] + w[1] * z[1]
            o = 1.0 if m >= 0 else -1.0
            sign = 1.0 if r > 0 else -1.0
            w = [w[j] + RATE * sign * z[j] for j in range(2)]
        else:
            o = 1.0 if rls.predict(z) >= 0 else -1.0
            rew.add(r)
            y = rew.z(r) if rew.n >= 2 else (1.0 if r > 0 else -1.0)
            rls.update(z, y)
            w = list(rls.w)
        nov.add(s_nov)
        prog.add(s_prog)
        k.outcome(o)
        captured, _lapsed = k.sleep(note="%s episode %d" % (method, i))
        kept += captured > 0
        if need_probe:
            before = bpb(k, probe) if captured else before
        log.append((i, len(ep), s_nov, s_prog, o, r,
                    list(w) if method in ("mixture", "adaptive")
                    else None))
    res = {"method": method, "final": bpb(k, final),
           "retention": bpb(k, episodes), "kept": kept, "log": log,
           "generation": k.meta.get("generation", 0),
           "cpu": time.process_time() - c0, "wall": time.time() - t0}
    return res


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 2
    know, corpus, outdir = argv[:3]
    salt = argv[4] if len(argv) > 4 and argv[3] == "--salt" else ""
    parts = {"train": [], "val": [], "test": []}
    for p, data in repo_files(corpus):
        parts[X.split(p, salt)].append(data)
    episodes = parts["val"]
    probe, final = parts["test"][0::2], parts["test"][1::2]
    print("episodes %d files %d bytes; probe %d files %d bytes; final %d "
          "files %d bytes" % (
              len(episodes), sum(map(len, episodes)), len(probe),
              sum(map(len, probe)), len(final), sum(map(len, final))))
    k0 = Knowledge(know)
    base_final = bpb(k0, final)
    base_ret = bpb(k0, episodes)
    print("generation 0: final %.6f, episodes %.6f bits per byte" % (
        base_final, base_ret))
    t0 = time.time()
    specs = [(m, know, outdir, episodes, probe, final) for m in METHODS]
    with multiprocessing.get_context("fork").Pool(len(METHODS)) as pool:
        results = pool.map(run, specs, chunksize=1)
    wall = time.time() - t0
    print("\nper episode: method, episode, bytes, surprise, progress, "
          "outcome, external reward, mixture weights")
    for r in results:
        for i, n, sn, sp, o, rew, w in r["log"]:
            print("  %-11s %2d %6d  %.3f  %+.3f  %+g  %s  %s" % (
                r["method"], i, n, sn, sp, o,
                "%+.2e" % rew if rew is not None else "-",
                "[%+.2f %+.2f]" % tuple(w) if w else ""))
    print("\nbits per byte (lower is better); final half scored by no "
          "method")
    print("  %-11s %9s %10s %11s %6s %8s" % (
        "method", "final", "vs gen 0", "episodes", "kept", "cpu s"))
    for r in results:
        print("  %-11s %9.6f %+10.6f %11.6f %3d/%-2d %8.1f" % (
            r["method"], r["final"], r["final"] - base_final,
            r["retention"], r["kept"], len(episodes), r["cpu"]))
    cpu = sum(r["cpu"] for r in results)
    print("\nresources: wall %.0f s, CPU %.0f core-s on %d processes; "
          "reading runs on the CPU (core/py/locus/knowledge.py)"
          % (wall, cpu, len(METHODS)))
    for m in METHODS:
        base = os.path.join(outdir, m, "store")
        for ext in ("", ".json", ".prev", ".prev.json"):
            if os.path.exists(base + ext):
                print("wrote %s  %d bytes" % (base + ext,
                                              os.path.getsize(base + ext)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
