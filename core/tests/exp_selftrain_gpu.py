"""Self-training on the device: does babble earn its place in the corpus?

  exp_selftrain_gpu.py STORE CORPUS BINARY [--babble-bytes N]
                       [--seeds K] [--salt S] [--jobs J]

The CPU version (exp_selftrain.py) takes a trained store and reads a
little more into it. That is the biologically faithful shape -- an
organism does not retrain from birth -- but it can only afford kilobytes
against a corpus of megabytes, and a difference of 0.001 bits at that
ratio cannot be told from noise.

This asks the question the device can actually answer: TRAIN FROM
SCRATCH on corpus + babble, and compare with corpus alone and with
corpus + an equal quantity of real text. Same three arms, same held-out
judge, but the added material is a measurable fraction of the whole
rather than a rounding error.

WHAT IS ON WHICH PROCESSOR, AND WHY:

  babble generation -- CPU, and unavoidably serial along the sequence,
  since each byte conditions on the one before it. Nothing about that
  parallelises. But SEEDS are independent, so K seeds run on K cores at
  the wall-clock cost of one, which is also how this gets an error bar
  instead of a single number.

  training and scoring -- device, one invocation per arm through the
  same `device_run` that builds every store in this project. The arms
  are serialised deliberately: they would contend for one GPU context
  and the failure mode for that is a wedged driver, not a slow run.

REPORTED AS MEAN AND SPREAD ACROSS SEEDS. One seed is an anecdote: the
first CPU run showed self +0.001290 with no way to say whether that was
the babble or the sampler's luck.

Nothing writes to STORE; it is read to generate, and the arms build
their own stores in a temporary directory.
"""

import os
import random
import shutil
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "py"))

from locus.knowledge import Knowledge          # noqa: E402
from locus.encode import NgramEncoder, repo_files   # noqa: E402
import exp_babble as B                         # noqa: E402
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
import babble_gpu as BG                        # noqa: E402
import exp_learn_gpu as G                      # noqa: E402
import exp_learn_stream as X                   # noqa: E402

LEARNER = "cls, neocortex also metaplastic (1/n)"


def one_babble(args):
    """`count` bytes from `store` at `seed`. Its own process."""
    store, count, seed = args
    know = Knowledge(store)
    return B.babble(know, count, b"Once upon a time",
                    random.Random(seed))


def run_arm(binary, enc, train, val, test, extra):
    """Device train+score on train+extra; returns test bpb at best lam."""
    files = list(train) + ([extra] if extra else [])
    tmp = tempfile.mkdtemp(prefix="selftrain-gpu-")
    try:
        _lb, kind, opts = [t for t in G.LEARNERS if t[0] == LEARNER][0]
        vb, tb, _nv, _nt = G.device_run(binary, enc, files, val, test,
                                        kind, opts,
                                        know=os.path.join(tmp, "k"),
                                        gated=True)
        _val, lam = min(zip(vb, G.LAMS))
        return tb[G.LAMS.index(lam)], lam
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def spread(values):
    """mean and half-range -- K is small, so quote the range, not a sd."""
    return sum(values) / len(values), (max(values) - min(values)) / 2.0


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 2
    store, corpus, binary, rest = argv[0], argv[1], argv[2], argv[3:]
    count, seeds, salt, jobs = 1 << 20, 4, "", 0
    while rest:
        if rest[0] == "--babble-bytes" and len(rest) > 1:
            count, rest = int(rest[1]), rest[2:]
        elif rest[0] == "--seeds" and len(rest) > 1:
            seeds, rest = int(rest[1]), rest[2:]
        elif rest[0] == "--salt" and len(rest) > 1:
            salt, rest = rest[1], rest[2:]
        elif rest[0] == "--jobs" and len(rest) > 1:
            jobs, rest = int(rest[1]), rest[2:]
        else:
            print("unknown argument: %s" % rest[0])
            return 2
    jobs = jobs or min(seeds, os.cpu_count() or 1)

    parts = {"train": [], "val": [], "test": []}
    for path, data in repo_files(corpus):
        parts[X.split(path, salt)].append(data)
    enc = NgramEncoder(parts["train"])
    train_bytes = sum(map(len, parts["train"]))
    print("corpus    %s: %d train files, %d bytes"
          % (corpus, len(parts["train"]), train_bytes))
    print("babble    %d bytes x %d seeds (%.2f%% of train each)"
          % (count, seeds, 100.0 * count / max(1, train_bytes)))
    print("where     generation and the arms both on the device; "
          "the arms are serialised so they cannot contend for one "
          "GPU context")
    print()

    # GENERATION IS ON THE DEVICE TOO (tools/babble_gpu.py, verified
    # byte-for-byte against the CPU babbler). It used to be K processes
    # on K cores while the GPU sat idle for minutes; at 787,659 bytes/s
    # the whole set is seconds, so the seed count is now chosen for the
    # error bar rather than for what a core could afford.
    t0 = time.time()
    babbles = BG.generate(store, os.path.join(os.path.dirname(binary),
                                              "babble_device"),
                          seeds, count, b"Once upon a time")
    print("generated %d streams x %d bytes on the device in %.1f s"
          % (len(babbles), count, time.time() - t0))

    # Equal quantity of REAL text the store has not read, cut out of the
    # scored set so no arm can memorise part of its own exam.
    pool_text = b"".join(parts["test"])
    if len(pool_text) <= count * 2:
        print("held-out text too short to cut a control from")
        return 2
    control = pool_text[-count:]
    test = [pool_text[:-count]]

    base, _lam = run_arm(binary, enc, parts["train"], parts["val"],
                         test, None)
    print("baseline  %.6f bits per byte" % base)

    selves = []
    for i, bab in enumerate(babbles, 1):
        bpb, _lam = run_arm(binary, enc, parts["train"], parts["val"],
                            test, bab)
        selves.append(bpb)
        print("self   %d  %.6f  (%+.6f)" % (i, bpb, bpb - base))

    ctrl, _lam = run_arm(binary, enc, parts["train"], parts["val"],
                         test, control)
    print("control   %.6f  (%+.6f)" % (ctrl, ctrl - base))

    m, hr = spread(selves)
    print()
    print("self      mean %.6f +/- %.6f  (%+.6f vs baseline)"
          % (m, hr, m - base))
    print("control   %+.6f -- the upper bound for this many bytes"
          % (ctrl - base))
    print()
    print("A self effect smaller than the spread across seeds is the")
    print("sampler, not the babble.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
