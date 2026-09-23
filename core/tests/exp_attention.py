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
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
