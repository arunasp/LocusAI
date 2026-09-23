"""Score stores on fixed evaluation sets they never trained on.

  exp_curriculum.py --eval NAME=DIR [--eval NAME=DIR ...] STORE [STORE ...]

The curriculum question -- does reading simple English before complex
English beat reading the complex text alone, for the same bytes? -- needs
every arm judged on the SAME text. The split inside `make know` cannot do
that: it assigns train, validation and test by path hash, so two arms
built from different files hold out different material and their figures
are not comparable.

So the evaluation sets here are directories kept out of every arm's
training corpus, and each store reads them with learning OFF. Nothing is
written; the stores are safe to pass directly.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "py"))
sys.path.insert(0, os.path.dirname(__file__))

from locus.encode import repo_files            # noqa: E402
from locus.execution import map_jobs           # noqa: E402
from locus.knowledge import Knowledge          # noqa: E402

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
    evals, paths, rest = [], [], list(argv)
    while rest:
        if rest[0] == "--eval" and len(rest) > 1:
            name, _, path = rest[1].partition("=")
            evals.append((name, path))
            rest = rest[2:]
        elif rest[0].startswith("--"):
            print("unknown argument: %s" % rest[0])
            return 2
        else:
            paths.append(rest[0])
            rest = rest[1:]
    if not evals or not paths:
        print(__doc__)
        return 2

    sets = []
    for name, path in evals:
        files = [d for _, d in repo_files(path)]
        if not files:
            print("no files in %s" % path)
            return 2
        sets.append((name, files))
        print("eval %-16s %d files, %d bytes"
              % (name, len(files), sum(len(d) for d in files)))
    print()
    head = "%-22s %10s" % ("store", "trained")
    for name, _ in sets:
        head += " %14s" % name
    print(head)
    for p in paths:
        k = Knowledge(p)
        row = "%-22s %10d" % (os.path.basename(p),
                              k.meta.get("train_bytes", 0))
        for _, files in sets:
            bpb, _n = score(k, files)
            row += " %14.6f" % bpb
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
