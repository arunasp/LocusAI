"""Record an outcome (modulator) event at the store's current tick.

  outcome.py KNOW VALUE

At sleep, VALUE is credited to every tag set at or before this tick and
within the lifetime (see core/py/locus/knowledge.py).
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "py"))

from locus.knowledge import Knowledge  # noqa: E402


def main(argv):
    if len(argv) != 2:
        print(__doc__)
        return 2
    k = Knowledge(argv[0])
    k.outcome(float(argv[1]))
    k.save_transient()
    print("outcome   %+g at tick %d; %d rows in the transient tier"
          % (float(argv[1]), k.clock, k.tagged()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
