"""Does more experience mean recognising a mistake more sharply?

  exp_recognition.py CORPUS STORE [STORE ...] [--files N] [--swap ab]

exp_recover.py showed a corrupted text reads far above normal text, so
a mistake announces itself as surprise. This asks whether that signal
GROWS with experience: a reader that knows more English should find
broken English more surprising, not less.

Each store reads the same held-out sample twice with learning OFF --
once clean, once with two letters swapped throughout (default e<->a,
which keeps the byte statistics and breaks the patterns) -- and the
margin between them is the recognition signal. Nothing is written, so
the stores are safe to pass directly.

Reported per store: its training size, clean bits per byte, corrupted
bits per byte, and the margin. A margin that widens with experience is
the prediction; a margin that flattens or narrows would say the signal
saturates, which matters for any plasticity rule that keys on it.
"""

import os
import sys

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
    out = bytearray(data)
    for i, c in enumerate(out):
        if c == a:
            out[i] = b
        elif c == b:
            out[i] = a
    return bytes(out)


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    corpus, paths, rest = argv[0], [], argv[1:]
    while rest and not rest[0].startswith("--"):
        paths.append(rest[0])
        rest = rest[1:]
    nfiles, swap = 4, "ea"
    while rest:
        if rest[0] == "--files" and len(rest) > 1:
            nfiles, rest = int(rest[1]), rest[2:]
        elif rest[0] == "--swap" and len(rest) > 1:
            swap, rest = rest[1], rest[2:]
        else:
            print("unknown argument: %s" % rest[0])
            return 2
    if not paths:
        print("name at least one store")
        return 2
    a, b = ord(swap[0]), ord(swap[1])

    # The SAME held-out sample for every store, so the stores differ only
    # in what they have read. The split comes from the first store's
    # salt; a store trained on a different salt would be reading its own
    # training text and is refused rather than quietly compared.
    first = Knowledge(paths[0])
    salt = first.meta["salt"]
    sample = [d for p, d in repo_files(corpus)
              if X.split(p, salt) == "test"][:nfiles]
    wrong = [corrupt(d, a, b) for d in sample]
    print("sample    %d held-out files, %d bytes; swapping %r and %r"
          % (len(sample), sum(len(d) for d in sample), swap[0], swap[1]))
    print()
    print("%-26s %10s %10s %10s %8s"
          % ("store", "trained", "clean", "corrupted", "margin"))
    for path in paths:
        k = Knowledge(path)
        if k.meta["salt"] != salt:
            print("%-26s SKIPPED: different salt, its test split differs"
                  % path)
            continue
        clean = score(k, sample)
        broken = score(k, wrong)
        print("%-26s %10d %10.6f %10.6f %8.4f"
              % (os.path.basename(path), k.meta.get("train_bytes", 0)
                 or k.meta.get("train_files", 0), clean, broken,
                 broken - clean))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
