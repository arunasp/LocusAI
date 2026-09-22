"""Learning next-byte structure from real text with the three-factor rule.

Stream: the repository's text files (`locus.encode.repo_files`), split by
a stable path hash into train, validation and test files. The encoder is
sized on the training files. The learners and the readout live in
`locus.learn`; this script holds only the split, the baselines and the
lam fit.

Per training position t, with context units C = units_at(data, t):

  1. Predict: drive[b] = sum over u in C of w[u -> b], for byte units b.
     p(b) = (max(drive[b], 0) + lam / 256) / (sum of positive drive + lam)
  2. Surprise s = -log2 p(next byte).
  3. Tag C -> next byte (`observe_transition`), then `consolidate(s)`:
     learning is largest where the prediction was worst.
  4. At each file end (the slow period), rescale every source row touched
     since the last rescale to sum 1 (`scale_sources`). Next bytes
     compete for a source's fixed total; nothing is ever depressed.

lam is the one free readout parameter. During training it only scales
the learning signal and is 1, as the reference's add-one. For scoring
it is fitted on the validation files, never on the test files.

Reference (not LocusAI): per-unit next-byte counts summed with add-one
smoothing, as in exp_stream.py, on the same split. Control: the same
counts with every unit's row scaled to sum 1, the equal weight per source
that `scale_sources` imposes, with lam fitted on validation. The last
learner variant must reproduce it exactly (see VARIANTS).

Run: python3 tests/exp_learn_stream.py [ROOT [GROUPS]]
(or `make stream-learn`). GROUPS is a comma-separated subset of tag,
faithful, branch and cls; all run when it is omitted. The reference and the
control always run.
"""

import math
import os
import sys
import time
import zlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "py"))

from locus.encode import BYTE_UNITS, NgramEncoder, repo_files  # noqa: E402
from locus.learn import (BranchLearner, CLSLearner,            # noqa: E402
                         FaithfulLearner, TagLearner, prob, score,
                         score_with)
from locus.plasticity import Plasticity                         # noqa: E402

EVERY = 5

# Ablation of the learner: (label, trace_decay, surprise-modulated,
# rescale at every file end). A decay of 1e-5 drops every tag below the
# floor before the next position, so only the current context is
# credited. The last variant removes all three differences from the
# control (and the weight cap), so its weights are proportional to counts
# and it must reproduce the control: that checks the mechanics.
VARIANTS = (
    ("learned, surprise, carry 0.7", 0.7, True, True),
    ("learned, constant, carry 0.7", 0.7, False, True),
    ("learned, surprise, no carry", 1e-5, True, True),
    ("learned, constant, no carry", 1e-5, False, True),
    ("check: constant, no carry, one rescale, no cap", 1e-5, False, False),
)

# The biology-faithful learner and its ablation: (label, NE gain,
# homeostasis). See locus.learn.FaithfulLearner.
FAITHFUL = (
    ("faithful: delta, NE gain, homeostasis", True, True),
    ("faithful without NE gain", False, True),
    ("faithful without homeostasis", True, False),
)

# Per-input errors: (label, metaplastic, NE gain). See
# locus.learn.BranchLearner. The metaplastic learner's rows are running
# frequencies, so it must reproduce the control.
BRANCH = (
    ("branch, constant rate", False, False),
    ("check: branch, metaplastic (1/n)", True, False),
    ("branch, metaplastic, NE gain", True, True),
)

# Complementary learning systems: (label, hippocampus, replay). See
# locus.learn.CLSLearner. The last is the cortex alone without replay,
# which must reproduce "branch, constant rate".
CLS = (
    ("cls: hippocampus, cortex, replay", True, True),
    ("cls without replay", True, False),
    ("check: cls cortex alone, no replay", False, False),
)


def split(path, salt=""):
    """train, val or test for ``path``; ``salt`` gives another split."""
    h = zlib.crc32((salt + path).encode())
    if h % EVERY == 0:
        return "test"
    if (h // EVERY) % EVERY == 0:
        return "val"
    return "train"


def count_rows(enc, train):
    counts = {}
    for data in train:
        for pos in range(len(data) - 1):
            nxt = data[pos + 1]
            for u in enc.units_at(data, pos):
                row = counts.setdefault(u, [0] * (BYTE_UNITS + 1))
                row[nxt] += 1
                row[BYTE_UNITS] += 1
    return counts


def score_normalised(counts, enc, files, lam):
    """Control: the count reference with every unit's row scaled to sum
    1, the same equal weight per source that `scale_sources` imposes."""
    bits = 0.0
    n = 0
    for data in files:
        for pos in range(len(data) - 1):
            drive = {}
            for u in enc.units_at(data, pos):
                row = counts.get(u)
                if row:
                    t = row[BYTE_UNITS]
                    for b in range(BYTE_UNITS):
                        if row[b]:
                            drive[b] = drive.get(b, 0.0) + row[b] / t
            bits -= math.log2(prob(drive, data[pos + 1], lam))
            n += 1
    return bits / max(n, 1)


def control_parts(counts, enc, files):
    """The lam-independent part of the control readout, per position:
    (drive of the actual byte, clipped at 0; sum of positive drive).
    Computed once, so a lam fit costs one pass instead of one per lam;
    `control_bpb` then gives the same figures as `score_normalised`."""
    dnext, dpos = [], []
    for data in files:
        for pos in range(len(data) - 1):
            drive = {}
            for u in enc.units_at(data, pos):
                row = counts.get(u)
                if row:
                    t = row[BYTE_UNITS]
                    for b in range(BYTE_UNITS):
                        if row[b]:
                            drive[b] = drive.get(b, 0.0) + row[b] / t
            dnext.append(max(drive.get(data[pos + 1], 0.0), 0.0))
            dpos.append(sum(v for v in drive.values() if v > 0.0))
    return dnext, dpos


def control_bpb(parts, lam):
    """Bits per byte from `control_parts`, as `prob` computes them."""
    dnext, dpos = parts
    bits = 0.0
    for dn, dp in zip(dnext, dpos):
        bits -= math.log2((dn + lam / BYTE_UNITS) / (dp + lam))
    return bits / max(len(dnext), 1)


def fit_lam(scorer, lo=-12, hi=4, limit=64):
    """lam on a grid of powers of two, EXTENDED until the best value is
    interior: a minimum sitting on a boundary means the grid, not the
    data, chose it. Returns (score, lam, at_edge); at_edge is true only
    if `limit` extensions were not enough, which is a harness bound, not
    a fitted value."""
    fits = {k: scorer(2.0 ** k) for k in range(lo, hi + 1)}
    for _ in range(limit):
        k = min(fits, key=lambda x: (fits[x], x))
        # extend only while the boundary is still IMPROVING outward: a
        # plateau means the grid is already wide enough, and extending
        # on a tie would walk off without ever finding anything better.
        if k == lo and fits[lo] < fits[lo + 1]:
            lo -= 1
            fits[lo] = scorer(2.0 ** lo)
        elif k == hi and fits[hi] < fits[hi - 1]:
            hi += 1
            fits[hi] = scorer(2.0 ** hi)
        else:
            return fits[k], 2.0 ** k, False
    k = min(fits, key=lambda x: (fits[x], x))
    return fits[k], 2.0 ** k, k in (lo, hi)


def reference(counts, enc, test):
    bits = 0.0
    n = 0
    for data in test:
        for pos in range(len(data) - 1):
            nxt = data[pos + 1]
            hit, total = 1, BYTE_UNITS
            for u in enc.units_at(data, pos):
                row = counts.get(u)
                if row:
                    hit += row[nxt]
                    total += row[BYTE_UNITS]
            bits -= math.log2(hit / total)
            n += 1
    return bits / max(n, 1)


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "..", "..")
    parts = {"train": [], "val": [], "test": []}
    for path, data in repo_files(root):
        parts[split(path)].append(data)
    for k in ("train", "val", "test"):
        size = sum(map(len, parts[k]))
        print("%-5s %3d files %8d bytes" % (k, len(parts[k]), size))
    if not all(parts.values()):
        print("INVALID: every split needs at least one file")
        return 2
    enc = NgramEncoder(parts["train"])
    print("units: %d" % enc.n)

    learned = []
    edge = False
    known = {"tag", "faithful", "branch", "cls"}
    groups = set(sys.argv[2].split(",")) if len(sys.argv) > 2 else known
    unknown = groups - known
    if unknown:
        print("INVALID: unknown group(s) %s" % ", ".join(sorted(unknown)))
        return 2
    for label, decay, modulated, per_file in (
            VARIANTS if "tag" in groups else ()):
        t0 = time.time()
        p = Plasticity(trace_decay=decay,
                       w_max=5.0 if per_file else float("inf"))
        online = TagLearner(p, enc, modulated, per_file).learn(
            parts["train"])
        val_bpb, lam, e = fit_lam(
            lambda x: score(p, enc, parts["val"], x)[0])
        edge = edge or e
        test_bpb, n = score(p, enc, parts["test"], lam)
        learned.append((label, test_bpb))
        print("%-47s %4.0f s  %7d weights  online %.3f  lam %-7g val %.3f%s"
              % (label, time.time() - t0, p.live_weights(), online, lam,
                 val_bpb, "  AT GRID EDGE" if e else ""))

    for label, ne_gain, homeostasis in (
            FAITHFUL if "faithful" in groups else ()):
        t0 = time.time()
        p = Plasticity()
        online = FaithfulLearner(p, enc, ne_gain, homeostasis).learn(
            parts["train"])
        val_bpb, lam, e = fit_lam(
            lambda x: score(p, enc, parts["val"], x)[0])
        edge = edge or e
        test_bpb, n = score(p, enc, parts["test"], lam)
        learned.append((label, test_bpb))
        print("%-47s %4.0f s  %7d weights  online %.3f  lam %-7g val %.3f%s"
              % (label, time.time() - t0, p.live_weights(), online, lam,
                 val_bpb, "  AT GRID EDGE" if e else ""))

    for label, metaplastic, ne_gain in (
            BRANCH if "branch" in groups else ()):
        t0 = time.time()
        p = Plasticity()
        online = BranchLearner(p, enc, metaplastic, ne_gain).learn(
            parts["train"])
        val_bpb, lam, e = fit_lam(
            lambda x: score(p, enc, parts["val"], x)[0])
        edge = edge or e
        test_bpb, n = score(p, enc, parts["test"], lam)
        learned.append((label, test_bpb))
        print("%-47s %4.0f s  %7d weights  online %.3f  lam %-7g val %.3f%s"
              % (label, time.time() - t0, p.live_weights(), online, lam,
                 val_bpb, "  AT GRID EDGE" if e else ""))

    for label, hippocampus, replay in (CLS if "cls" in groups else ()):
        t0 = time.time()
        cls = CLSLearner(Plasticity(), Plasticity(), enc, replay=replay,
                         hippocampus=hippocampus)
        online = cls.learn(parts["train"])
        val_bpb, lam, e = fit_lam(
            lambda x: score_with(cls.drive, enc, parts["val"], x)[0])
        edge = edge or e
        test_bpb, n = score_with(cls.drive, enc, parts["test"], lam)
        learned.append((label, test_bpb))
        weights = cls.hippo.live_weights() + cls.cortex.live_weights()
        print("%-47s %4.0f s  %7d weights  online %.3f  lam %-7g val %.3f%s"
              % (label, time.time() - t0, weights, online, lam, val_bpb,
                 "  AT GRID EDGE" if e else ""))

    counts = count_rows(enc, parts["train"])
    cparts = control_parts(counts, enc, parts["val"])
    cval, clam, cedge = fit_lam(lambda x: control_bpb(cparts, x))
    print("control lam fitted on validation: %g (val %.3f)%s" % (
        clam, cval, "  AT GRID EDGE -- widen the grid" if cedge else ""))

    ref = reference(counts, enc, parts["test"])
    ctrl = score_normalised(counts, enc, parts["test"], clam)
    print("\nheld-out test, bits per byte (%d predictions):" % n)
    print("  %-47s %.3f" % ("uniform", 8.0))
    print("  %-47s %.3f" % ("reference (count, add-one)", ref))
    print("  %-47s %.3f" % ("control (count, rows sum to 1)", ctrl))
    for label, bpb in learned:
        print("  %-47s %.3f" % (label, bpb))
    return 2 if edge or cedge else 0


if __name__ == "__main__":
    raise SystemExit(main())
