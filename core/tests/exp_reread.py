"""What a SECOND reading of the same text does.

  exp_reread.py STORE CORPUS [--files N] [--passes N]

Reads the same train files repeatedly with learning on, and scores the
held-out test split before and after. Three questions, each answered by
a number rather than by the theory:

1. Does the text itself get cheaper to read on the second pass? That is
   recognition of this text, and the metaplastic 1/n rate says the gain
   must shrink with every repetition.
2. Does held-out text get cheaper? That is generalisation, and
   re-reading adds no new information, so it should barely move.
3. Does any of it survive? Reading only TAGS rows in the transient
   tier; sleep keeps the tagged rows that outcomes credit and lapses the
   rest, so a re-reading with no outcome should leave the permanent
   store as it was.

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
    return bits / max(n, 1), n


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    path, corpus, rest = argv[0], argv[1], argv[2:]
    nfiles, passes = 20, 3
    while rest:
        if rest[0] == "--files" and len(rest) > 1:
            nfiles, rest = int(rest[1]), rest[2:]
        elif rest[0] == "--passes" and len(rest) > 1:
            passes, rest = int(rest[1]), rest[2:]
        else:
            print("unknown argument: %s" % rest[0])
            return 2

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
    print("store     %s (generation %d)"
          % (path, k.meta.get("generation", 0)))
    print("re-read   %d train files, %d bytes; test split %d files"
          % (len(train), sum(len(d) for d in train), len(test)))

    before, npred = score(k, test)
    print("held-out  %.9f bits per byte over %d predictions (before)"
          % (before, npred))

    print()
    print("pass   bits per byte on the SAME text   gain   wall")
    prev = None
    for i in range(1, passes + 1):
        t0 = time.time()
        bits = n = 0
        for data in train:
            b = k.read(data, learn=True)
            bits += sum(b)
            n += len(b)
        got = bits / max(n, 1)
        print("%-6d %-33.9f %-6s %.1f s"
              % (i, got, "%+.6f" % (got - prev) if prev else "-",
                 time.time() - t0))
        prev = got

    after, _ = score(k, test)
    print()
    print("held-out  %.9f bits per byte (after)   change %+.9f"
          % (after, after - before))
    print("store     generation %d, %d rows tagged in the transient tier"
          % (k.meta.get("generation", 0), k.tagged()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
