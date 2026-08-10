"""Post-deploy smoke test: exercise the built artifact's real ABI.

Distinct from the unit suites, which test behaviour. This confirms the
compiled shared library loads and round-trips through the ctypes
binding at all -- the closest equivalent to running a deployed binary.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "py"))

from locus import Pathway, Store, Tier  # noqa: E402


def main():
    with Store() as store:
        if not store.put(1, "ok", pathway=Pathway.INSTINCT):
            raise SystemExit("FAIL: put rejected")
        lease = store.lookup(1)
        if lease is None:
            raise SystemExit("FAIL: lookup missed")
        try:
            if lease.data != b"ok":
                raise SystemExit("FAIL: payload %r" % lease.data)
        finally:
            lease.release()
        if store.tier(1) is not Tier.ACTIVE:
            raise SystemExit("FAIL: instinct trace not resident")
    print("PASS: verify")


if __name__ == "__main__":
    main()
