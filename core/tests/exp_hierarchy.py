"""What compression ratio survives, measured before the loop is built.

  exp_hierarchy.py [--n N] [--cues K] [--ratios A,B,C] [--jobs J]

doc/core/HIERARCHY.md is design-only, and it names the two quantities
that decide the level operator rather than leaving the ratio to
choice:

  SEPARABILITY   distinct level-0 attractors map to distinct upper
                 supports
  RECOVERABILITY an upper support, fed back, settles level 0 into the
                 attractor that produced it

"Compression is the ratio at which separability and recoverability
still hold. It is never the ratio alone."

Neither needs the settle loop. A candidate projection over the
existing Field answers both: feedforward W is local receptive fields
sized from n/m with overlap from the kernel's reach (Decision 2, where
n/m is the only chosen quantity), and feedback is its transpose
entering level 0 as an input rather than a clamp (Projections).

WHAT THIS DOES NOT MEASURE, and it matters for reading the result: no
weights are learned. The design refines both projections with the
three-factor rule on co-active settled sets, so these figures are the
FLOOR an untrained projection gives, not the ceiling a trained one
reaches. A ratio that already fails untrained is the interesting
answer; one that passes says only that learning starts from somewhere
workable.

Ratios run in parallel, one process each -- they share nothing.
"""

import os
import sys
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "py"))
sys.path.insert(0, HERE)

from locus.field import Field                  # noqa: E402
from fixtures import random_kernel             # noqa: E402
from watchdog import PASS, Budget, stopped_on_event  # noqa: E402


def settle(f, budget=24):
    """Cycle until the support STOPS CHANGING, or the cap is reached.

    A CAP IS NOT A STOPPING CRITERION, and this file had the error the
    rule was written for: a fixed 12 cycles per run, with separability
    compared across ratios from the results. HIERARCHY.md states the
    requirement -- every run reports its termination event, and a sweep
    containing a cap-terminated run is not compared -- and
    `watchdog.Budget` exists in this repo to make the difference
    impossible to leave implicit.

    Returns (support, settled): `settled` is True when the support
    repeated, False when the budget ran out.
    """
    b = Budget(budget, "settle")
    previous = None
    while True:
        f.cycle()
        support = support_of(f)
        if support == previous:
            return support, True
        previous = support
        if not b.spend():
            return support, False


def support_of(f):
    """Indices above the field's OWN mean, as a set.

    Two traps, both paid for. `Field.active(threshold)` returns (index,
    value) PAIRS -- read as indices it makes every support a set of
    distinct tuples, so nothing intersects and every overlap reads 0.00
    while every support reads distinct. And the 1e-12 floor admits
    EVERY unit (measured: 128 of 128), so a floor-based support is the
    whole field and says nothing. The mean is a rank, not a chosen
    threshold, and it is the same rule residency uses.
    """
    mean = sum(f.a) / len(f.a)
    return frozenset(i for i, v in enumerate(f.a) if v > mean)


def attractors(n, cues, seed):
    """Settled supports from distinct single-unit cues.

    The support -- which units are above the floor -- is what a
    projection sees, so it is what separability is about.
    """
    kernel = random_kernel(n, seed=seed)
    out = []
    for c in range(cues):
        f = Field(n, kernel, seed=seed)
        f.inject(c * (n // cues), 1.0)
        support, settled = settle(f)
        out.append((c * (n // cues), support, list(f.a), settled))
    return kernel, out


def project(state, n, m, reach):
    """Upper activity from local receptive fields, no learning.

    Coverage is about n/m lower units per upper unit and overlap
    follows the kernel's reach, which is Decision 2 exactly.
    """
    width = max(1, n // m) + reach
    upper = []
    for u in range(m):
        centre = int((u + 0.5) * n / m)
        lo, hi = max(0, centre - width // 2), min(n, centre + width // 2)
        upper.append(sum(state[lo:hi]))
    return upper


def one_ratio(args):
    n, cues, ratio, seed, reach, fb = args
    m = max(1, n // ratio)
    kernel, settled = attractors(n, cues, seed)

    uppers = []
    for _cue, _support, state, _ok in settled:
        up = project(state, n, m, reach)
        # An upper unit counts as in the support when it is above the
        # pool's OWN mean -- the same rank rule the store uses for
        # residency, and no threshold of its own.
        mean_up = sum(up) / len(up) if up else 0.0
        uppers.append(frozenset(i for i, v in enumerate(up)
                                if v > mean_up))

    distinct_low = len({s for _c, s, _st, _ok in settled})
    distinct_up = len(set(uppers))
    separable = distinct_up / max(1, distinct_low)

    # Recoverability: feed an upper support back as INPUT (not a clamp)
    # and see whether level 0 settles into the support that produced it.
    # OVERLAP, NOT A PASS RATE. "Recovered" needs a threshold, and a
    # threshold here would be a number chosen to make the answer come
    # out. The overlap itself is the measurement; CHANCE is the overlap
    # between DIFFERENT attractors, which is what it has to beat.
    overlaps = []
    recover_ok = []
    for (cue, support, _state, _ok), up in zip(settled, uppers):
        f = Field(n, kernel, seed=seed)
        width = max(1, n // m) + reach
        for u in up:
            centre = int((u + 0.5) * n / m)
            for i in range(max(0, centre - width // 2),
                           min(n, centre + width // 2)):
                f.inject(i, fb / width)
        back, back_ok = settle(f)
        recover_ok.append(back_ok)
        if back and support:
            overlaps.append(len(back & support) / len(back | support))
    chance = []
    sets = [sup for _c, sup, _st, _ok in settled]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            if sets[i] and sets[j]:
                chance.append(len(sets[i] & sets[j])
                              / len(sets[i] | sets[j]))
    mean = sum(overlaps) / len(overlaps) if overlaps else 0.0
    base = sum(chance) / len(chance) if chance else 0.0
    flags = [ok for _c, _s, _st, ok in settled] + recover_ok
    verdict, why = stopped_on_event(flags)
    return (ratio, m, distinct_low, distinct_up, separable, mean, base,
            verdict, why)


def main(argv):
    n, cues, seed, reach, jobs = 128, 8, 11, 4, 0
    # Feedback amplitude, swept rather than assumed: it enters level 0
    # as an INPUT, so too little moves nothing and too much clamps.
    fb = 1.0
    ratios = [2, 4, 8, 16, 32]
    while argv:
        if argv[0] == "--n" and len(argv) > 1:
            n, argv = int(argv[1]), argv[2:]
        elif argv[0] == "--cues" and len(argv) > 1:
            cues, argv = int(argv[1]), argv[2:]
        elif argv[0] == "--ratios" and len(argv) > 1:
            ratios, argv = [int(x) for x in argv[1].split(",")], argv[2:]
        elif argv[0] == "--seed" and len(argv) > 1:
            seed, argv = int(argv[1]), argv[2:]
        elif argv[0] == "--fb" and len(argv) > 1:
            fb, argv = float(argv[1]), argv[2:]
        elif argv[0] == "--jobs" and len(argv) > 1:
            jobs, argv = int(argv[1]), argv[2:]
        else:
            print("unknown argument: %s" % argv[0])
            return 2
    jobs = jobs or min(len(ratios), os.cpu_count() or 1)

    print("n %d, %d cues, seed %d, untrained projection" % (n, cues, seed))
    print("%6s %6s %10s %10s %12s %14s"
          % ("n/m", "m", "low sets", "up sets", "separable", "overlap"))
    work = [(n, cues, r, seed, reach, fb) for r in ratios]
    with Pool(jobs) as pool:
        for row in pool.map(one_ratio, work):
            ratio, m, dl, du, sep, mean, base, verdict, why = row
            print("%6d %6d %10d %10d %11.0f%% %8.2f vs %.2f chance  %s"
                  % (ratio, m, dl, du, 100 * sep, mean, base, verdict))
            if verdict != PASS:
                print("       %s -- this row is not comparable" % why)
    print()
    print("Compression is the ratio at which BOTH still hold. Untrained,")
    print("so a ratio that already fails here is the informative case.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
