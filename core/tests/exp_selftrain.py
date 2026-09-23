"""Does learning from its own babble change what the store knows?

  exp_selftrain.py STORE CORPUS [--bytes N] [--sample-seed T] [--salt S]

Stage 2's actual question. Babbling produces a stream; this measures
whether CONSUMING that stream does anything, against the only judge
that matters -- held-out text the store has never read.

THREE ARMS, all scored on the same held-out split with the same reader:

  baseline   the store as trained
  self       the store after reading its own babble
  control    the store after reading an EQUAL NUMBER OF BYTES of real
             held-out text

The control is what makes the result interpretable. Without it, "self
made it worse" cannot be told apart from "any further reading at this
point makes it worse", and "self made it better" cannot be told apart
from "more bytes of anything help". The control reads held-out text,
which is genuinely new information, so it is the upper bound on what
any equal quantity of reading could buy -- and the self arm brings no
new information by construction, since it came out of the store.

PREDICTED BEFORE RUNNING, kept because a prediction is only worth
something beside what happened: self should be neutral to harmful,
because sampling then learning re-weights the store toward what it
already favours and no information enters.

MEASURED: harmful at every scale. ts-32 +0.001290 against a control of
+0.000065; ts-512 +0.000073 against +0.000000. The device version
(exp_selftrain_gpu.py) settles it with a spread -- ts-2002 +0.002657
+/- 0.000009 across 8 seeds, control -0.000059.

WHAT THAT LICENSES, STATED NO WIDER THAN THE MEASUREMENT. Babble is
imprecise by construction -- it is what the store believes, sampled --
so its use is as a MEASUREMENT OF HOW WELL A TRAINING STEP WENT:
exp_babble.py reads production directly, where bits per byte only
reads prediction, and the two moved differently across the learning
curve.

What is ruled out is feeding it back UNCONTROLLED. Nothing here says
self-generated material can never help; it says an unfiltered stream
does not, because sampling and re-learning re-weights the store toward
what it already favoured and no information enters. A feedback path
would need a controlled stream -- selected, checked, or scored against
something outside the store -- and that control is the part that would
have to be built and measured, not assumed.

Nothing is written to the original store: each arm loads its own copy.
"""

import os
import random
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "py"))

from locus.knowledge import Knowledge          # noqa: E402
import exp_babble as B                         # noqa: E402
import exp_learn_stream as X                   # noqa: E402


def held_out(corpus, salt=""):
    """Every test-split file of the corpus, concatenated."""
    parts = []
    for name in sorted(os.listdir(corpus)):
        path = os.path.join(corpus, name)
        if os.path.isfile(path) and X.split(path, salt) == "test":
            with open(path, "rb") as fh:
                parts.append(fh.read())
    return b"".join(parts)


def copy_store(path):
    """A private copy of the store, so an arm cannot touch the original."""
    tmp = tempfile.mkdtemp(prefix="selftrain-")
    out = os.path.join(tmp, os.path.basename(path))
    for suffix in ("", ".json", ".tags"):
        if os.path.exists(path + suffix):
            shutil.copy(path + suffix, out + suffix)
    return tmp, out


def score(know, text):
    """Bits per byte on `text`, read without learning."""
    bits = know.read(text, learn=False)
    return sum(bits) / max(1, len(bits))


def arm(store, text, test):
    """bpb on `test` after reading `text` (None: read nothing)."""
    tmp, path = copy_store(store)
    try:
        know = Knowledge(path)
        if text:
            know.read(text, learn=True)
        return score(know, test)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    store, corpus, rest = argv[0], argv[1], argv[2:]
    count, sample_seed, salt = 20000, 1, ""
    while rest:
        if rest[0] == "--bytes" and len(rest) > 1:
            count, rest = int(rest[1]), rest[2:]
        elif rest[0] == "--sample-seed" and len(rest) > 1:
            sample_seed, rest = int(rest[1]), rest[2:]
        elif rest[0] == "--salt" and len(rest) > 1:
            salt, rest = rest[1], rest[2:]
        else:
            print("unknown argument: %s" % rest[0])
            return 2

    test = held_out(corpus, salt)
    if not test:
        print("no held-out files in %s" % corpus)
        return 2

    # The babble comes from an untouched copy: generating must not be
    # able to change the store the arms start from.
    tmp, path = copy_store(store)
    try:
        speaker = Knowledge(path)
        rng = random.Random(sample_seed)
        babble = B.babble(speaker, count, b"Once upon a time", rng)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # THE CONTROL'S MATERIAL IS CUT OUT OF THE SCORED SET. Reading the
    # tail of the held-out text and then scoring the whole of it lets
    # the control memorise part of its own exam, which showed up as a
    # spurious improvement the first time this ran. The scored text is
    # everything except the tail; the control reads the tail; the self
    # arm is scored on exactly the same text so the arms stay
    # comparable.
    if len(test) <= count * 2:
        print("held-out text too short to hold out a control from")
        return 2
    real = test[-count:]
    test = test[:-count]

    print("store     %s" % store)
    print("held-out  %d bytes from %s" % (len(test), corpus))
    print("babble    %d bytes, sample seed %d" % (len(babble), sample_seed))
    print("control   %d bytes of real held-out text" % len(real))
    print()

    base = arm(store, None, test)
    self_ = arm(store, babble, test)
    ctrl = arm(store, real, test)

    print("baseline  %.6f bits per byte" % base)
    print("self      %.6f  (%+.6f)" % (self_, self_ - base))
    print("control   %.6f  (%+.6f)" % (ctrl, ctrl - base))
    print()
    print("Negative is better. The control is the upper bound on what")
    print("this many bytes of reading can buy; the self arm carries no")
    print("information the store did not already hold.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
