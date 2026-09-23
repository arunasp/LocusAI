"""How far back does anything actually reach? Measured, per window.

  exp_attention.py STORE [--bytes N] [--max-distance D] [--seed S]

doc/core/ATTENTION.md defines five windows from biology and records
that the effective window LocusAI has is UNMEASURED. This measures the
two that can be measured today, and they are different quantities on
different paths.

READOUT REACH (the knowledge path). `Knowledge.drive` reads
`NgramEncoder.units_at`, so the prediction at a position can only
depend on the bytes those n-grams cover -- orders (2, 3, 4) means the
current byte and the three before it. That is a claim about the code,
so it is testable: corrupt the byte at distance d and the bits charged
must change for d within reach and NOT CHANGE AT ALL beyond it. A
measured reach longer than the orders would mean the reader is reading
something the encoder did not give it.

PERSISTENCE (the store path). A store carries activation, heat and an
active set across ticks, none of which the readout uses. So influence
there can outlast the n-grams, and the question is how many periods a
cue survives -- the sampling window's practical length. Measured as
the overlap between the active set right after a cue and the active
set k ticks later, with no further input.

The two are reported apart because conflating them is exactly the
mistake ATTENTION.md warns about: the readout's reach is a span in
bytes, the store's persistence is a count of periods.

Standard library only, CPU only. Nothing is written to the store.
"""

import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "py"))

from locus.knowledge import Knowledge          # noqa: E402
from locus import store as S                   # noqa: E402


def readout_reach(know, data, max_d, rng):
    """Bits charged at each position, and the same with the byte at
    distance d corrupted. Returns [(d, mean |change|, changed positions)].

    Only positions far enough into the text to HAVE a byte at distance d
    are compared, so a short prefix cannot look like an effect.
    """
    base = know.read(data, learn=False)
    out = []
    for d in range(1, max_d + 1):
        edit = bytearray(data)
        hits = []
        # One corruption per comparison, spaced so two edits never fall
        # inside one another's reach.
        for pos in range(d + max_d, len(data) - 1, max_d * 4):
            old = edit[pos - d]
            new = rng.randrange(256)
            if new == old:
                new = (new + 1) % 256
            edit[pos - d] = new
            hits.append(pos)
        after = know.read(bytes(edit), learn=False)
        # read() charges the byte AFTER each position, so index pos-1.
        deltas = [abs(after[p - 1] - base[p - 1]) for p in hits
                  if p - 1 < len(base)]
        changed = sum(1 for x in deltas if x > 1e-12)
        mean = sum(deltas) / len(deltas) if deltas else 0.0
        out.append((d, mean, changed, len(deltas)))
    return out


def persistence(cue_keys, ticks, capacity=64, active_k=8):
    """Share of the cued set still ACTIVE k ticks later, no further
    input. A count of periods, not of bytes.

    `capacity` 64 and `active_k` 8 are BOUNDS on the measurement, not
    claims about the substrate: the store has to be larger than the cue
    set for the cue set to be what is measured, and active_k has to be
    at least the cue count or kWTA -- not persistence -- decides the
    answer. Cowan's four-item figure (doc/core/ATTENTION.md) is about
    the capacity window and is measured elsewhere.

    There is no active-set accessor, so the set is read as the keys
    whose tier is ACTIVE -- the same thing enforce_kwta decides.
    """
    with S.Store(capacity=capacity, active_k=active_k) as st:
        for k in cue_keys:
            st.put(k, b"c", S.Pathway.DECLARATIVE, 1.0)
            st.excite(k, 2.0)
        st.tick()
        first = {k for k in cue_keys if st.tier(k) == S.Tier.ACTIVE}
        out = []
        for k in range(1, ticks + 1):
            st.tick()
            now = {j for j in cue_keys if st.tier(j) == S.Tier.ACTIVE}
            keep = len(first & now) / max(1, len(first))
            act = [st.activation(j) for j in cue_keys]
            out.append((k, keep, len(now), max(act) if act else 0.0))
    return out


def capacity(counts, active_k=64, capacity_slots=256):
    """How many traces stay ACTIVE as the cued population grows.

    ATTENTION.md's capacity window is ~4 items limited by INTERFERENCE
    rather than by storage (Cowan 2001), and records that `active_k` is
    configured here instead. This measures which of the two is binding:
    with active_k set well above every cue count, an interference limit
    would show as the active count LEVELLING OFF, and a configured
    bound would show as the active count tracking the cue count.

    `active_k` 64 and 256 slots are BOUNDS chosen so neither can be the
    limit being measured.
    """
    out = []
    for n in counts:
        with S.Store(capacity=capacity_slots, active_k=active_k) as st:
            for i in range(1, n + 1):
                st.put(i, b"c", S.Pathway.DECLARATIVE, 1.0)
                st.excite(i, 2.0)
            st.tick()
            act = sum(1 for i in range(1, n + 1)
                      if st.tier(i) == S.Tier.ACTIVE)
            peak = max(st.activation(i) for i in range(1, n + 1))
        out.append((n, act, peak))
    return out


def return_bias(drive, ticks_away, capacity_slots=64):
    """Does a RECENTLY ATTENDED trace come back easier or harder than a
    fresh one at the same drive?

    ATTENTION.md's refractory window: biology disfavours a location just
    attended (inhibition of return, Posner & Cohen 1984). Nothing here
    implements it, and `heat` outliving activation suggests the
    OPPOSITE sign -- a recently active trace should be easier to
    revive. Sign first: build nothing until the direction is measured.

    A is cued, wins, then goes quiet for `ticks_away`. Then A and a
    never-seen B are cued with identical drive. Whose activation and
    tier come out ahead is the answer.
    """
    out = []
    for away in ticks_away:
        with S.Store(capacity=capacity_slots, active_k=4) as st:
            st.put(1, b"a", S.Pathway.DECLARATIVE, 1.0)
            st.excite(1, drive)
            st.tick()
            for _ in range(away):
                st.tick()
            st.put(2, b"b", S.Pathway.DECLARATIVE, 1.0)
            st.excite(1, drive)
            st.excite(2, drive)
            st.tick()
            out.append((away, st.activation(1), st.activation(2),
                        S.Tier(st.tier(1)).name,
                        S.Tier(st.tier(2)).name))
    return out


def two_rates(drive, ticks, beta=0.1):
    """Activation and heat after one cue, and what a heat-keyed input
    depression would do to a later revisit.

    The refractory window is biphasic in biology: detection is BETTER
    for ~100-300 ms after attending and WORSE from ~500-3000 ms, and
    the mechanism sits on the input -- short-term depression of early
    sensory input (Satel et al. 2011), habituation of the orienting
    response (Dukewich 2009). The trace is not weakened; what would
    re-orient to it is.

    Two decay rates are therefore all the shape needs: activation gives
    the early advantage, heat outlives it and gives the later cost. This
    measures both, and the crossover a `drive/(1 + beta*heat)`
    depression would produce -- so the coefficient is chosen against
    numbers rather than by borrowing one that was handy.

    `beta` 0.1 here is the candidate coefficient BEING EVALUATED, not a
    setting: it is the store's lateral-competition value, tried on the
    input path to see whether it produces a crossover in the right
    place. Measured: it does not -- 9% depression against a residual
    advantage starting at +2.37, crossing at about tick 19 where
    biology's facilitation-to-inhibition ratio is 2-10x. Nothing here
    is wired into the store.
    """
    out = []
    with S.Store(capacity=32, active_k=4) as st:
        st.put(1, b"a", S.Pathway.DECLARATIVE, 1.0)
        st.excite(1, drive)
        for k in range(1, ticks + 1):
            st.tick()
            act, heat = st.activation(1), st.heat(1)
            depressed = drive / (1.0 + beta * heat)
            # A revisit is worth residual activation plus a depressed
            # drive; a fresh trace is worth the undepressed drive.
            out.append((k, act, heat, depressed,
                        act + depressed - drive))
    return out


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    store, rest = argv[0], argv[1:]
    count, max_d, seed = 60000, 10, 1
    while rest:
        if rest[0] == "--bytes" and len(rest) > 1:
            count, rest = int(rest[1]), rest[2:]
        elif rest[0] == "--max-distance" and len(rest) > 1:
            max_d, rest = int(rest[1]), rest[2:]
        elif rest[0] == "--seed" and len(rest) > 1:
            seed, rest = int(rest[1]), rest[2:]
        else:
            print("unknown argument: %s" % rest[0])
            return 2

    know = Knowledge(store)
    rng = random.Random(seed)
    text = (b"Once upon a time there was a little girl who loved to "
            b"play outside with her dog. ") * (count // 100 + 1)
    text = text[:count]

    print("store     %s, orders %s" % (store, list(know.enc.orders)))
    print("text      %d bytes" % len(text))
    print()
    print("READOUT REACH -- bits charged when the byte at distance d is")
    print("corrupted. Zero beyond the encoder's reach is the claim.")
    print("%6s %14s %12s %8s" % ("d", "mean |dbits|", "changed", "of"))
    for d, mean, changed, n in readout_reach(know, text, max_d, rng):
        print("%6d %14.6f %12d %8d" % (d, mean, changed, n))
    print()
    print("PERSISTENCE -- share of the cued active set still active k")
    print("ticks later, no further input.")
    print("%6s %10s %8s %12s" % ("ticks", "kept", "active", "max act"))
    for k, keep, n, peak in persistence(list(range(1, 9)), 8):
        print("%6d %9.2f%% %8d %12.4f" % (k, 100.0 * keep, n, peak))
    print()
    print("CAPACITY -- active traces as the cued population grows, with")
    print("active_k and slots set high enough that neither binds.")
    print("%6s %10s %12s" % ("cued", "active", "max act"))
    for n, act, peak in capacity([1, 2, 4, 8, 16, 32]):
        print("%6d %10d %12.4f" % (n, act, peak))
    print()
    print("RETURN -- a recently attended trace against a fresh one at")
    print("the same drive. Biology disfavours the revisited one.")
    print("%6s %12s %12s %10s %10s"
          % ("away", "revisited", "fresh", "rev tier", "new tier"))
    for away, a1, a2, t1, t2 in return_bias(2.0, [1, 2, 4, 8]):
        print("%6d %12.4f %12.4f %10s %10s" % (away, a1, a2, t1, t2))
    print()
    print("TWO RATES -- what a heat-keyed input depression would buy.")
    print("Positive advantage means revisiting is still EASIER.")
    print("%5s %11s %9s %11s %11s"
          % ("tick", "activation", "heat", "depressed", "advantage"))
    for k, act, heat, dep, adv in two_rates(2.0, 12):
        print("%5d %11.4f %9.4f %11.4f %+11.4f"
              % (k, act, heat, dep, adv))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
