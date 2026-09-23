"""Babbling: read the store's own expectations back out as bytes.

  exp_babble.py STORE [--bytes N] [--seed S] [--sample-seed T] [--show N]

Stage 2 of the developmental path. Every measurement before this one
fed LocusAI someone else's text and asked how well it predicted the
next byte. This asks the opposite question -- what does it produce when
nothing is feeding it -- and the answer is a property of what it
learned, not of the corpus.

THE SAMPLING IS THE STORE'S OWN DISTRIBUTION, with no temperature and
no top-k. A temperature would be a constant nobody measured sitting
between the store and its output, and it would make the babble a
property of that constant rather than of the store; sampling
proportionally is what the model actually believes. The only reshaping
here is `--seed`, the context the first bytes are drawn from, which is
the equivalent of an infant hearing a sound before producing one.

WHAT IS REPORTED, and why each number is here rather than a judgement
about whether the output "looks like English":

  real-word rate -- the share of whitespace-separated tokens that occur
  in the corpus the store read. A store that has learned nothing above
  the letter produces letter salad and scores near zero; one that has
  learned words scores high without any claim about grammar.

  novel-word rate -- the share of DISTINCT tokens that do NOT occur in
  the corpus. Both extremes are failures: zero means it is replaying,
  high means it is inventing noise. Reported, not scored.

  self-bits -- what the store charges for its own output. Compare it
  with what the same store charges for real held-out text: far lower
  means it produces only what it is most certain of, which is the
  collapse this experiment exists to detect.

Standard library only, CPU only. The store is read, never written.
"""

import os
import random
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "py"))

from locus.knowledge import Knowledge          # noqa: E402
from locus.encode import BYTE_UNITS            # noqa: E402
import exp_learn_stream as X                   # noqa: E402


def babble(know, count, seed_text, rng):
    """`count` bytes sampled from the store's own next-byte drive."""
    buf = bytearray(seed_text)
    for _ in range(count):
        drive = know.drive(bytes(buf), len(buf) - 1)
        total = sum(drive.values()) + know.lam
        # The same expression `expect()` reports and `bits()` charges:
        # sampling has to read the distribution the scorer uses, or the
        # babble measures a second, unrelated model.
        r = rng.random() * total
        acc = 0.0
        pick = None
        for b in range(BYTE_UNITS):
            acc += drive.get(b, 0.0) + know.lam / BYTE_UNITS
            if acc >= r:
                pick = b
                break
        buf.append(pick if pick is not None else BYTE_UNITS - 1)
    return bytes(buf[len(seed_text):])


def babble_many(store_path, count, seeds, seed_text=b"Once upon a time",
                jobs=None):
    """`count` bytes per seed, one process per seed.

    THE CPU PATH IS NOT A LEFTOVER. A stream is serial along its own
    sequence, but seeds are independent, so they are the parallel
    dimension on whichever processor is available: the device carries
    them as a batch (tools/babble_gpu.py), and here they are one
    process each. This is what runs where there is no device binary --
    a boot without a GPU, a worker container, or while the device is
    held by another job -- and it is the reference the kernel is
    verified against, so it cannot be allowed to rot.
    """
    jobs = jobs or min(len(seeds), os.cpu_count() or 1)
    args = [(store_path, count, seed, seed_text) for seed in seeds]
    with Pool(jobs) as pool:
        return pool.map(_one, args)


def _one(args):
    """One stream, in its own process. Module level because a Pool
    pickles by name."""
    store_path, count, seed, seed_text = args
    return babble(Knowledge(store_path), count, seed_text,
                  random.Random(seed))


def vocabulary(corpus_dir, held_out=True, salt=""):
    """The tokens of the corpus, from the HELD-OUT split by default.

    Scoring against the whole corpus rewards a large store for
    reproducing text it read: the same words are in its training set and
    in the reference, so the rate rises with exposure whether or not
    anything was learned about word FORM. The test split is the same
    English and none of it was read, so a token found there is evidence
    about form rather than recall. `--whole-corpus` restores the old
    reference for comparison, and the two are reported apart.
    """
    vocab = set()
    for name in sorted(os.listdir(corpus_dir)):
        path = os.path.join(corpus_dir, name)
        if not os.path.isfile(path):
            continue
        if held_out and X.split(path, salt) != "test":
            continue
        with open(path, "rb") as fh:
            for tok in fh.read().split():
                vocab.add(tok.lower().strip(b".,!?\"';:()"))
    return vocab


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    store, rest = argv[0], argv[1:]
    count, seed_text, sample_seed, show = 2000, b"Once upon a time", 1, 400
    corpus, held_out = None, True
    while rest:
        if rest[0] == "--bytes" and len(rest) > 1:
            count, rest = int(rest[1]), rest[2:]
        elif rest[0] == "--seed" and len(rest) > 1:
            seed_text, rest = rest[1].encode(), rest[2:]
        elif rest[0] == "--sample-seed" and len(rest) > 1:
            sample_seed, rest = int(rest[1]), rest[2:]
        elif rest[0] == "--show" and len(rest) > 1:
            show, rest = int(rest[1]), rest[2:]
        elif rest[0] == "--corpus" and len(rest) > 1:
            corpus, rest = rest[1], rest[2:]
        elif rest[0] == "--whole-corpus":
            held_out, rest = False, rest[1:]
        else:
            print("unknown argument: %s" % rest[0])
            return 2

    know = Knowledge(store)
    rng = random.Random(sample_seed)
    out = babble(know, count, seed_text, rng)

    print("store     %s (%d lifetime events, lam %g)"
          % (store, know.lifetime, know.lam))
    print("seed      %r" % seed_text.decode(errors="replace"))
    print("babble    %d bytes, sample seed %d" % (len(out), sample_seed))
    print()
    print(out[:show].decode(errors="replace"))
    print()

    # What the store charges for its own output, against what it charges
    # for the seed text it was given.
    bits = know.read(seed_text + out, learn=False)
    own = sum(bits[len(seed_text):]) / max(1, len(out))
    print("self-bits %.6f bits per byte on its own output" % own)

    if corpus:
        vocab = vocabulary(corpus, held_out)
        toks = [t.lower().strip(b".,!?\"';:()") for t in out.split()]
        toks = [t for t in toks if t]
        real = sum(1 for t in toks if t in vocab)
        distinct = set(toks)
        novel = sum(1 for t in distinct if t not in vocab)
        print("tokens    %d, %d distinct" % (len(toks), len(distinct)))
        print("reference %s split of %s"
              % ("held-out test" if held_out else "whole", corpus))
        print("real      %.1f%% of tokens occur in the reference"
              % (100.0 * real / max(1, len(toks))))
        print("novel     %.1f%% of distinct tokens do not"
              % (100.0 * novel / max(1, len(distinct))))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
