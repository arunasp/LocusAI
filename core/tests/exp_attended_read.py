"""Read English through the attention substrate, and score it.

  exp_attended_read.py STORE CORPUS [--bytes N] [--capacity C]

The two halves of this project have been separate all day. The English
figures come from encoder -> learn_device -> sparse store -> Python
reader, which never calls locus_put, locus_excite or anything in
store.c. The five attention windows live in the substrate, which never
sees English. Nothing measured on one side says anything about the
other, and that was true without being stated.

THIS IS THE JOIN, in its smallest measurable form. Both readers see
the same bytes and the same learned rows; they differ only in WHICH
units get to speak:

  reader     every unit units_at() returns, summed and gated
  attended   the units that are ACTIVE in the store after a tick

So the attended reader exercises the capacity window (how many units
hold a slot), the refractory depression (a unit used a moment ago is
driven less) and the pathway pools -- against bits per byte on the
same text, which is the measure every English figure here uses.

WHAT A RESULT MEANS. Worse is the expected outcome and not a failure:
the plain reader uses all four orders at every position, the attended
one uses a subset, and a subset of evidence predicts worse unless the
selection is buying something. The number worth having is HOW MUCH
worse, because that is the price of attention on this path, and it is
the thing to reduce rather than a verdict on the idea.

Standard library only, CPU, nothing written to the store on disk.
"""

import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "py"))
sys.path.insert(0, HERE)

from locus.knowledge import Knowledge, BYTE_UNITS   # noqa: E402
from locus import store as S                        # noqa: E402
import exp_learn_stream as X                        # noqa: E402


def held_out(corpus, limit, salt=""):
    parts = []
    for name in sorted(os.listdir(corpus)):
        path = os.path.join(corpus, name)
        if os.path.isfile(path) and X.split(path, salt) == "test":
            with open(path, "rb") as fh:
                parts.append(fh.read())
        if sum(map(len, parts)) >= limit:
            break
    return b"".join(parts)[:limit]


def bits_from(drive, nxt, lam):
    pos = sum(v for v in drive.values() if v > 0.0)
    p = (max(drive.get(nxt, 0.0), 0.0) + lam / BYTE_UNITS) / (pos + lam)
    return -math.log2(p) if p > 0.0 else 64.0


def plain_bits(know, data):
    return know.read(data, learn=False)


def attended_bits(know, data, capacity):
    """Same rows, but only the units the store keeps active speak."""
    out = []
    seen = set()
    with S.Store(capacity=capacity, active_k=capacity // 4) as st:
        for pos in range(len(data) - 1):
            units = know.enc.units_at(data, pos)
            live = [u for u in units if u >= 0]
            for u in live:
                if u not in seen:
                    st.put(u, b"u", S.Pathway.DECLARATIVE, 1.0)
                    seen.add(u)
                # Solicited: the task asked for this unit, so it is
                # exempt from the refractory depression. Reading is
                # recurrence and depressing it penalises the signal.
                st.attend(u, 1.0)
            st.tick()
            active = [u for u in live
                      if st.tier(u) == S.Tier.ACTIVE]
            rows = []
            for s in know.stores:
                for u in active:
                    rows.append(s.row(u))
            drive = {}
            for r in rows:
                for b, w in r.items():
                    drive[b] = drive.get(b, 0.0) + w
            out.append((bits_from(drive, data[pos + 1], know.lam),
                        len(active), len(live)))
    return out


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    store, corpus, rest = argv[0], argv[1], argv[2:]
    count, capacity = 20000, 256
    while rest:
        if rest[0] == "--bytes" and len(rest) > 1:
            count, rest = int(rest[1]), rest[2:]
        elif rest[0] == "--capacity" and len(rest) > 1:
            capacity, rest = int(rest[1]), rest[2:]
        else:
            print("unknown argument: %s" % rest[0])
            return 2

    know = Knowledge(store)
    data = held_out(corpus, count)
    print("store     %s (lam %g, orders %s)"
          % (store, know.lam, list(know.enc.orders)))
    print("text      %d held-out bytes, store capacity %d"
          % (len(data), capacity))
    print()

    plain = plain_bits(know, data)
    plain_bpb = sum(plain) / max(1, len(plain))
    rows = attended_bits(know, data, capacity)
    att_bpb = sum(r[0] for r in rows) / max(1, len(rows))
    active = sum(r[1] for r in rows) / max(1, len(rows))
    offered = sum(r[2] for r in rows) / max(1, len(rows))

    print("plain reader     %.6f bits per byte" % plain_bpb)
    print("attended reader  %.6f  (%+.6f)" % (att_bpb, att_bpb - plain_bpb))
    print("units active     %.2f of %.2f offered per position"
          % (active, offered))
    print()
    print("Worse is expected: the attended reader speaks with a subset")
    print("of the evidence. The figure to reduce is the gap.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
