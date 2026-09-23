"""Two ways to ask what repertoire a field has, and how they differ.

  exp_repertoire.py [--n N] [--seed S] [--stream K]

`Cycle.survey` ENUMERATES: it injects one unit at a time and counts the
stable supports that result. `Cycle.survey_driven` DRIVES: it replays a
stream of cues and counts the supports experience actually carved. The
second exists because the first is a procedure no organism performs --
nothing in a real substrate visits every unit in turn -- and it was
written and then never used, which `make audit` reported.

This uses it, and reports what the two instruments say about the same
field:

  enumerated   supports found by injecting each unit in turn
  driven       supports found by replaying a cue stream of the same size
  shared       supports both instruments found
  only-driven  supports the enumerating instrument never produced

A driven count far below the enumerated one means the enumeration
reports states that experience does not reach; anything in only-driven
means the enumeration misses states experience does reach. Both are
worth knowing before a repertoire figure is quoted anywhere.

The stream is drawn from a seeded generator, so the comparison is
reproducible; cues repeat, as experience does.
"""

import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "py"))
sys.path.insert(0, os.path.dirname(__file__))

from locus.cycle import Cycle                  # noqa: E402


def random_kernel(n, seed, deg=6):
    """The kernel exp_capacity.py uses, so the repertoire compared here
    is the one those measurements are about.

    A uniform ring was tried first and gave ONE support to both
    instruments: with nothing to distinguish regions the field has a
    single global attractor, and a comparison of two ways to count one
    thing says nothing.
    """
    rnd = random.Random(seed)
    rows = []
    for i in range(n):
        row = [0.0] * n
        row[i] = 0.3 + 0.2 * rnd.random()
        for _ in range(deg):
            row[rnd.randrange(n)] += rnd.random()
        total = sum(row)
        rows.append([v / total for v in row])
    return rows


def main(argv):
    n, seed, stream_len = 64, 71, None
    rest = list(argv)
    while rest:
        if rest[0] == "--n" and len(rest) > 1:
            n, rest = int(rest[1]), rest[2:]
        elif rest[0] == "--seed" and len(rest) > 1:
            seed, rest = int(rest[1]), rest[2:]
        elif rest[0] == "--stream" and len(rest) > 1:
            stream_len, rest = int(rest[1]), rest[2:]
        else:
            print(__doc__)
            return 2
    # Same number of injections for both instruments, so the comparison
    # is of the METHOD and not of how much each was allowed to look.
    stream_len = stream_len if stream_len is not None else n

    cycle = Cycle(n, random_kernel(n, seed))
    rng = random.Random(seed)
    stream = [rng.randrange(n) for _ in range(stream_len)]

    t0 = time.time()
    supports, unstable = cycle.survey()
    t_enum = time.time() - t0
    enumerated = {frozenset(s) for s in supports}

    t0 = time.time()
    found, unstable_driven = cycle.survey_driven(stream)
    t_driven = time.time() - t0
    driven = set(found)

    # The repeating stream visits FEWER UNIQUE cues than the enumeration
    # does (birthday collisions), so a gap between them mixes method
    # with coverage -- the confound survey_saturating's own docstring
    # warns about. This arm drives every cue exactly once, in a random
    # order, so method is the only difference left.
    every = list(range(n))
    random.Random(seed + 1).shuffle(every)
    covered, unstable_covered = cycle.survey_driven(every)
    matched = set(covered)

    # Both arms above take the sample count from me: n cues, because I
    # said so. survey_saturating takes it from the field instead -- it
    # draws until the repertoire stops growing and REPORTS how many that
    # took. That is the quantity the two arms above silently assumed.
    draw = random.Random(seed + 2)
    sat, unstable_sat, consumed, saturated = cycle.survey_saturating(
        lambda: draw.randrange(n))
    saturating = set(sat)

    print("field       n=%d, random kernel, seed %d" % (n, seed))
    print("enumerated  %d supports, %d unstable, %.2f s"
          % (len(enumerated), unstable, t_enum))
    print("driven      %d supports, %d unstable, %.2f s over %d cues "
          "(%d unique)"
          % (len(driven), unstable_driven, t_driven, len(stream),
             len(set(stream))))
    print("driven all  %d supports, %d unstable, over %d cues each once"
          % (len(matched), unstable_covered, len(every)))
    print("saturating  %d supports, %d unstable, %d cues drawn, %s"
          % (len(saturating), unstable_sat, consumed,
             "saturated" if saturated else "HIT THE CAP, not saturated"))
    print("shared      %d" % len(enumerated & driven))
    print("only driven %d" % len(driven - enumerated))
    print("only enum   %d" % len(enumerated - driven))
    if enumerated:
        print("driven/enumerated %.2f (repeating stream), %.2f (every cue "
              "once)" % (len(driven) / len(enumerated),
                         len(matched) / len(enumerated)))
    print("sat vs enum %d only in saturating, %d only in enumerated"
          % (len(saturating - enumerated), len(enumerated - saturating)))
    print("input share %.3f (drive arriving from outside at the end; "
          "a field running free decays toward zero)"
          % cycle.field.observed_input_share())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
