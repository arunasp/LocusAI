"""Does the hierarchy stack, and does it keep stacking as n grows?

WHAT THIS SETTLED. At n=64 the stack reached one level (4.6x) and then
refused with "only 1 distinct chunk", and I concluded that levels do not
stack -- which was wrong. At n=256 it reaches two levels and 85.3x,
terminating because the top FITS THE WINDOW rather than because a level
failed. The level-1 collapse was small-field, not structural, and the
conclusion drawn from it came from a field too small to have state at
level 1.

DEPTH AND COMPRESSION ARE BOTH OUTPUTS. Reaching 10^6 down to 10^3 was
once estimated as "about five levels" by dividing by a measured 4x, and
that arithmetic is wrong in principle. Compression moves with inhibition
(8.00x at beta 0.3, 4.27x at 0.5, 4.00x above), with anatomy (a random
kernel gives one attractor per state and therefore NO compression where
a ring gives 4x), and with n (8.00x then 10.67x at n=256 against 4.00x
at n=64). No constant survives that, so depth is discovered by driving
the stack until it fits, never computed.

THE DENSITY CONSTRAINT, measured rather than assumed. A trace chain sums
over every excursion, so the coarse kernel is far denser than the fine
one: 7.0 nonzeros per row at n=256 becomes 18.6 per row at n=32, 58%
dense. That does not prevent stacking -- level 1 still yielded 3 chunks
while 58% dense -- but it sets a FLOOR ON LEVEL SIZE, because a level
must be several times the coarse fan-out to hold distinguishable states.
At n=16 it yielded 1 and the stack stopped.

Run: python3 tests/exp_stack.py   (or `make stack`)
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "py"))
sys.path.insert(0, os.path.dirname(__file__))

import watchdog as wd                          # noqa: E402
from locus.cycle import Cycle                  # noqa: E402
from exp_operating_point import ring_kernel    # noqa: E402


def density(kernel):
    """Fraction of entries carrying mass, and fan-out per row."""
    n = len(kernel)
    if not n:
        return 0.0, 0.0
    nz = sum(1 for row in kernel for v in row if v > 1e-9)
    frac = nz / (n * n)
    return frac, frac * n


def run_stack(n, beta=1.0, window=8, seed=909, episodes=8,
              max_fast=80, stride=8, max_levels=6):
    kernel = ring_kernel(n, seed)
    c = Cycle(n, kernel, beta=beta)
    seeds = [i for i in range(0, n, max(1, n // 8))]
    t0 = time.time()
    levels = c.drive_and_stack(evidence=seeds, window=window,
                               max_levels=max_levels,
                               episodes=episodes, max_fast=max_fast,
                               stride=stride)
    elapsed = time.time() - t0
    comps = [r["compression"] for r in levels
             if r["compression"] is not None]
    cumulative = 1.0
    for x in comps:
        cumulative *= x
    return {
        "levels": levels,
        "comps": comps,
        "cumulative": cumulative,
        "depth": len(comps),
        "stop": levels[-1]["stop"] if levels else None,
        "elapsed": elapsed,
        "fine_density": density(kernel),
        "cycle": c,
    }


def main():
    print("=== does it stack, and how deep, as n grows? ===")
    print("  n     depth  per-level compressions        cumulative  "
          "secs  stop")
    depths, cums = [], []
    first_comps = []
    for n in (64, 128, 256, 512):
        r = run_stack(n)
        depths.append(float(r["depth"]))
        cums.append(r["cumulative"])
        if r["comps"]:
            first_comps.append(r["comps"][0])
        shown = ", ".join("%.2fx" % x for x in r["comps"]) or "-"
        print("  %-5d %-6d %-29s %-11.1f %-5.0f %s"
              % (n, r["depth"], shown[:29], r["cumulative"],
                 r["elapsed"], (r["stop"] or "")[:34]))

    print()
    print("=== checks on the sweep ===")
    for name, vals in (("depth vs n", depths),
                       ("cumulative vs n", cums),
                       ("level-0 compression vs n", first_comps)):
        status, detail = wd.param_has_effect(vals)
        print("  [%s] %-26s %s" % (status, name, detail))
    status, detail = wd.monotone(cums, tol=0.5)
    print("  [%s] %-26s %s" % (status, "cumulative monotone", detail))

    print()
    print("=== the density floor on level size ===")
    print("  n     fine_fanout  coarse_n  coarse_fanout  coarse_frac")
    for n in (128, 256, 512):
        r = run_stack(n)
        res = r["cycle"].level_up(max_fast=80, stride=8)
        if not res.admissible:
            print("  %-5d %-12.1f refused: %s"
                  % (n, r["fine_density"][1], res.reason[:40]))
            continue
        frac, fan = density(res.kernel)
        print("  %-5d %-12.1f %-9d %-14.1f %.3f"
              % (n, r["fine_density"][1], res.chunks, fan, frac))
    print("  -> a trace chain sums over every excursion, so the coarse")
    print("     kernel is denser than the fine one by construction. A")
    print("     level must exceed that fan-out by some factor to hold")
    print("     distinguishable states, which is the floor on level")
    print("     size and the reason a small level collapses to one.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
