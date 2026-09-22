"""Coincidence-gated readout: does a unit's row count more when it agrees
with the other active units?

  exp_match_gate.py CORPUS STORE[:SALT] [STORE[:SALT] ...]

For each store (generation 0, trained on its split), every validation and
test position is read out two ways:

- base: the sum of all active rows (the device's readout);
- gated: each (store, unit) row r is scaled by
  g = sigmoid(signed_sqrt(cos(r, D - r))), D the sum of all active rows,
  i.e. by its agreement with the consensus of the others (NMDA-like
  coincidence detection; the gate form of DeepSeek-V4.1's Engram).

The gate is recomputed at every position from the current rows; nothing
is tuned. lam is refitted on validation for each readout. The base
readout must reproduce the store's device test figure. One process per
(store, readout).
"""

import math
import multiprocessing
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "py"))
sys.path.insert(0, HERE)

import exp_learn_stream as X  # noqa: E402
from locus.encode import BYTE_UNITS, repo_files  # noqa: E402
from locus.knowledge import Knowledge  # noqa: E402

LAMS = [2.0 ** k for k in range(-12, 5)]


def gate(r, rest):
    """sigmoid(signed sqrt(cos(r, rest)))."""
    dot = sum(w * rest.get(b, 0.0) for b, w in r.items())
    nr = math.sqrt(sum(w * w for w in r.values()))
    nc = math.sqrt(sum(w * w for w in rest.values()))
    c = dot / (nr * nc) if nr > 0 and nc > 0 else 0.0
    s = math.copysign(math.sqrt(abs(c)), c)
    return 1.0 / (1.0 + math.exp(-s))


def parts(k, files, gated):
    """Per position (drive of the actual next byte, sum of positive
    drive), in the device's clip-then-sum form."""
    dn, dp = [], []
    for data in files:
        for pos in range(len(data) - 1):
            rows = [s.row(u) for s, units in k.active(data, pos)
                    for u in units]
            rows = [r for r in rows if r]
            total = {}
            for r in rows:
                for b, w in r.items():
                    total[b] = total.get(b, 0.0) + w
            if gated and len(rows) > 1:
                d = {}
                for r in rows:
                    rest = {b: total[b] - r.get(b, 0.0) for b in total}
                    g = gate(r, rest)
                    for b, w in r.items():
                        d[b] = d.get(b, 0.0) + g * w
                total = d
            pos_drive = {b: v for b, v in total.items() if v > 0.0}
            dn.append(pos_drive.get(data[pos + 1], 0.0))
            dp.append(sum(pos_drive.values()))
    return dn, dp


def bpb(p, lam):
    dn, dp = p
    bits = 0.0
    for a, b in zip(dn, dp):
        bits -= math.log2((a + lam / BYTE_UNITS) / (b + lam))
    return bits / max(len(dn), 1)


def job(spec):
    corpus, store, salt, gated = spec
    t0, c0 = time.time(), time.process_time()
    k = Knowledge(store)
    split = {"val": [], "test": []}
    for p, d in repo_files(corpus):
        s = X.split(p, salt)
        if s in split:
            split[s].append(d)
    vp = parts(k, split["val"], gated)
    val, lam = min((bpb(vp, x), x) for x in LAMS)
    test = bpb(parts(k, split["test"], gated), lam)
    return {"store": store, "salt": salt, "gated": gated, "val": val,
            "lam": lam, "test": test, "device": k.meta["test_bpb"],
            "cpu": time.process_time() - c0, "wall": time.time() - t0}


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    corpus = argv[0]
    specs = []
    for a in argv[1:]:
        store, _, salt = a.partition(":")
        specs += [(corpus, store, salt, g) for g in (False, True)]
    t0 = time.time()
    with multiprocessing.get_context("fork").Pool(len(specs)) as pool:
        res = pool.map(job, specs, chunksize=1)
    print("%-22s %-6s %9s %10s %9s %9s %9s" % (
        "store", "split", "readout", "lam", "val", "test", "vs base"))
    ok = True
    for i in range(0, len(res), 2):
        b, g = res[i], res[i + 1]
        ok &= abs(b["test"] - b["device"]) < 1e-9
        for r in (b, g):
            print("%-22s %-6s %9s %10g %9.6f %9.6f %+9.6f" % (
                os.path.basename(r["store"]), r["salt"] or "default",
                "gated" if r["gated"] else "base", r["lam"], r["val"],
                r["test"], r["test"] - b["test"]))
    print("base reproduces the device test figure on every store: %s"
          % ok)
    print("resources: wall %.0f s, CPU %.0f core-s on %d processes "
          "(Python readout on the CPU)" % (
              time.time() - t0, sum(r["cpu"] for r in res), len(res)))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
