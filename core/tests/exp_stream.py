"""Real input for learning: local repository text through the byte and
hashed n-gram encoder, with a reference next-byte score to beat.

The stream is every text file of the repository this runs in, read as
bytes (`locus.encode.repo_files`). About one file in `EVERY` is held
out, chosen by a stable hash of its path. The encoder is sized on the
training files only.

The reference predictor is not LocusAI: at each position it adds the
next-byte counts of every active unit, with add-one smoothing, and is
scored in bits per byte on the held-out files. It exists so a learned
result has a figure to beat. Uniform guessing is 8.0 bits per byte.

Run: python3 tests/exp_stream.py [ROOT]   (or `make stream`)
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "py"))

from locus.encode import (BYTE_UNITS, NgramEncoder, held_out,  # noqa: E402
                          repo_files)

EVERY = 5


def score(enc, train, test, use_ngrams):
    counts = {}
    for data in train:
        for pos in range(len(data) - 1):
            units = enc.units_at(data, pos)
            if not use_ngrams:
                units = units[:1]
            nxt = data[pos + 1]
            for u in units:
                row = counts.setdefault(u, [0] * (BYTE_UNITS + 1))
                row[nxt] += 1
                row[BYTE_UNITS] += 1
    bits = 0.0
    n = 0
    for data in test:
        for pos in range(len(data) - 1):
            units = enc.units_at(data, pos)
            if not use_ngrams:
                units = units[:1]
            nxt = data[pos + 1]
            hit = 1
            total = BYTE_UNITS
            for u in units:
                row = counts.get(u)
                if row:
                    hit += row[nxt]
                    total += row[BYTE_UNITS]
            bits -= math.log2(hit / total)
            n += 1
    return bits / n, n


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "..", "..")
    files = repo_files(root)
    train = [d for p, d in files if not held_out(p, EVERY)]
    test = [d for p, d in files if held_out(p, EVERY)]
    print("root %s: %d text files, %d train (%d bytes), %d held out "
          "(%d bytes)" % (os.path.abspath(root), len(files), len(train),
                          sum(map(len, train)), len(test),
                          sum(map(len, test))))
    if not train or not test:
        print("INVALID: need both training and held-out files")
        return 2
    enc = NgramEncoder(train)
    print("units: %d (256 byte + %d n-gram)" % (enc.n, enc.n - BYTE_UNITS))
    print("\norder head   size  distinct  sharing_a_slot")
    for k, h, size, distinct, shared in enc.collisions(train):
        print("%-5d %-4d %7d  %8d  %d" % (k, h, size, distinct, shared))
    print("\norder distinct  sharing_all_slots (indistinguishable)")
    for k, distinct, joint in enc.joint_collisions(train):
        print("%-5d %8d  %d" % (k, distinct, joint))
    print("\nreference, held-out bits per byte:")
    print("  uniform                     8.000")
    for label, use in (("byte context only", False),
                       ("byte + n-gram units", True)):
        bpb, n = score(enc, train, test, use)
        print("  %-27s %.3f   (%d predictions)" % (label, bpb, n))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
