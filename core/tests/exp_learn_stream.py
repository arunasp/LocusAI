"""Learning next-byte structure from real text with the three-factor rule.

Stream: the repository's text files (`locus.encode.repo_files`), split by
a stable path hash into train, validation and test files. The encoder is
sized on the training files.

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

Run: python3 tests/exp_learn_stream.py [ROOT]   (or `make stream-learn`)
"""

import math
import os
import sys
import time
import zlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "py"))

from locus.encode import BYTE_UNITS, NgramEncoder, repo_files  # noqa: E402
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


def split(path):
    h = zlib.crc32(path.encode())
    if h % EVERY == 0:
        return "test"
    if (h // EVERY) % EVERY == 0:
        return "val"
    return "train"


def drive_of(p, units):
    drive = {}
    for u in units:
        for b in p._out.get(u, ()):
            if b < BYTE_UNITS:
                drive[b] = drive.get(b, 0.0) + p.weights[(u, b)]
    return drive


def prob(drive, b, lam):
    pos = sum(v for v in drive.values() if v > 0.0)
    return (max(drive.get(b, 0.0), 0.0) + lam / BYTE_UNITS) / (pos + lam)


def learn(p, enc, files, modulated=True, per_file=True):
    bits = 0.0
    n = 0
    for data in files:
        touched = set()
        for pos in range(len(data) - 1):
            units = enc.units_at(data, pos)
            nxt = data[pos + 1]
            s = -math.log2(prob(drive_of(p, units), nxt, 1.0))
            bits += s
            n += 1
            p.observe_transition([(u, 1.0) for u in units], [(nxt, 1.0)])
            p.consolidate(s if modulated else 1.0)
            touched.update(units)
        if per_file:
            p.scale_sources(1.0, sources=touched)
    if not per_file:
        p.scale_sources(1.0)
    return bits / max(n, 1)


def score(p, enc, files, lam):
    bits = 0.0
    n = 0
    for data in files:
        for pos in range(len(data) - 1):
            b = data[pos + 1]
            drive = drive_of(p, enc.units_at(data, pos))
            bits -= math.log2(prob(drive, b, lam))
            n += 1
    return bits / max(n, 1), n


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


def fit_lam(scorer):
    """lam on a grid of powers of two; returns (score, lam, at_edge)."""
    fits = [(scorer(2.0 ** k), 2.0 ** k) for k in range(-12, 5)]
    best, lam = min(fits)
    return best, lam, lam in (fits[0][1], fits[-1][1])


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
    for label, decay, modulated, per_file in VARIANTS:
        t0 = time.time()
        p = Plasticity(trace_decay=decay,
                       w_max=5.0 if per_file else float("inf"))
        online = learn(p, enc, parts["train"], modulated, per_file)
        val_bpb, lam, e = fit_lam(
            lambda x: score(p, enc, parts["val"], x)[0])
        edge = edge or e
        test_bpb, n = score(p, enc, parts["test"], lam)
        learned.append((label, test_bpb))
        print("%-47s %4.0f s  %7d weights  online %.3f  lam %-7g val %.3f%s"
              % (label, time.time() - t0, p.live_weights(), online, lam,
                 val_bpb, "  AT GRID EDGE" if e else ""))

    counts = count_rows(enc, parts["train"])
    cval, clam, cedge = fit_lam(
        lambda x: score_normalised(counts, enc, parts["val"], x))
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
