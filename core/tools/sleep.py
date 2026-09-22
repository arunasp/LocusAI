"""Offline consolidation of the transient tier into the permanent tier.

  sleep.py KNOW [--lifetime TICKS]

Each tagged row sums the outcomes logged at or after its tick and within
TICKS of it (default: no limit). A positive sum captures the row into a
new generation of KNOW (previous kept as KNOW.prev); anything else lets
it lapse. The transient tier is cleared.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "py"))

from locus.knowledge import Knowledge  # noqa: E402


def main(argv):
    if not argv or len(argv) not in (1, 3):
        print(__doc__)
        return 2
    lifetime = None
    if len(argv) == 3:
        if argv[1] != "--lifetime":
            print(__doc__)
            return 2
        lifetime = int(argv[2])
    k = Knowledge(argv[0])
    pending = len(k.outcomes)
    captured, lapsed = k.sleep(lifetime=lifetime, note="sleep")
    print("sleep     %d outcomes, lifetime %s" % (
        pending, lifetime if lifetime is not None else "unlimited"))
    print("          %d rows captured, %d lapsed" % (captured, lapsed))
    if captured:
        print("          generation %d; previous kept as %s.prev"
              % (k.meta["generation"], argv[0]))
    for p in (argv[0], argv[0] + ".json"):
        print("          %s  %d bytes" % (p, os.path.getsize(p)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
