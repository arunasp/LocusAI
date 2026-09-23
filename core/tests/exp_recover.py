"""Can a mistaken reading be corrected by reading the truth again?

  exp_recover.py STORE CORPUS [--files N] [--passes N] [--swap ab]

Re-reading correct text changes almost nothing (exp_reread.py), which is
right: there is nothing to fix. This asks the other half -- what happens
when the earlier reading WAS wrong.

A copy of the train text is corrupted by swapping two letters
throughout (default e<->a), which keeps the byte statistics but breaks
the patterns. The store reads the corrupted text with learning on, and
then reads the CORRECT text pass after pass. Held-out bits per byte is
measured at every stage, so three numbers come out:

  damage      how much the mistaken reading cost
  recovery    how much of that comes back per pass of the truth
  residue     what is still missing after the last pass

The metaplastic rate is 1/n over a unit's own history, so corrupting a
unit ADDS to its count and makes it LESS movable afterwards. Recovery
being slow or partial is therefore a measurement of rigidity, not a bug,
and it is the case the reconsolidation design in memory is meant to
address.

The store is used in place, so pass a COPY.
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "py"))
sys.path.insert(0, os.path.dirname(__file__))

from locus.encode import repo_files            # noqa: E402
from locus.execution import map_jobs           # noqa: E402
from locus.knowledge import Knowledge          # noqa: E402
import exp_learn_stream as X                   # noqa: E402

STORE = None


def score_one(data):
    b = STORE.read(data, learn=False)
    return sum(b), len(b)


def score(store, files):
    global STORE
    STORE = store
    parts = map_jobs(score_one, files)
    bits = sum(p[0] for p in parts)
    n = sum(p[1] for p in parts)
    return bits / max(n, 1)


def corrupt(data, a, b):
    """Swap two letters throughout: same bytes, broken patterns."""
    out = bytearray(data)
    for i, c in enumerate(out):
        if c == a:
            out[i] = b
        elif c == b:
            out[i] = a
    return bytes(out)


def read_all(k, files):
    bits = n = 0
    for data in files:
        got = k.read(data, learn=True)
        bits += sum(got)
        n += len(got)
    return bits / max(n, 1)


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    path, corpus, rest = argv[0], argv[1], argv[2:]
    nfiles, passes, swap = 8, 3, "ea"
    while rest:
        if rest[0] == "--files" and len(rest) > 1:
            nfiles, rest = int(rest[1]), rest[2:]
        elif rest[0] == "--passes" and len(rest) > 1:
            passes, rest = int(rest[1]), rest[2:]
        elif rest[0] == "--swap" and len(rest) > 1:
            swap, rest = rest[1], rest[2:]
        else:
            print("unknown argument: %s" % rest[0])
            return 2
    a, b = ord(swap[0]), ord(swap[1])

    k = Knowledge(path)
    salt = k.meta["salt"]
    train, test = [], []
    for p, d in repo_files(corpus):
        part = X.split(p, salt)
        if part == "train":
            train.append(d)
        elif part == "test":
            test.append(d)
    train = train[:nfiles]
    wrong = [corrupt(d, a, b) for d in train]
    print("store     %s (generation %d)"
          % (path, k.meta.get("generation", 0)))
    print("text      %d train files, %d bytes; swapping %r and %r"
          % (len(train), sum(len(d) for d in train), swap[0], swap[1]))

    base = score(k, test)
    print("held-out  %.9f before anything" % base)

    t0 = time.time()
    on_wrong = read_all(k, wrong)
    damaged = score(k, test)
    print("MISTAKE   read the corrupted text at %.6f bpb; held-out now "
          "%.9f (%+.6f) in %.1f s"
          % (on_wrong, damaged, damaged - base, time.time() - t0))

    print()
    print("pass   held-out        vs damaged   vs original   wall")
    for i in range(1, passes + 1):
        t0 = time.time()
        read_all(k, train)
        now = score(k, test)
        print("%-6d %-15.9f %-12s %-13s %.1f s"
              % (i, now, "%+.6f" % (now - damaged),
                 "%+.6f" % (now - base), time.time() - t0))
    print()
    print("store     generation %d, %d rows tagged (nothing captured "
          "without an outcome)"
          % (k.meta.get("generation", 0), k.tagged()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
