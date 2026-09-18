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


def run_stack(n, beta=1.0, window=None, seed=909, episodes=8,
              max_fast=80, stride=1, max_levels=8):
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
    print("=== how deep does it go, with NO window? ===")
    print("  The window is gone. Every earlier run stopped with 'fits")
    print("  the window' before reaching any limit, so maximum depth")
    print("  was never measured and cross-run compression compared")
    print("  endpoints I had chosen. Now the only stops are a level")
    print("  REFUSING or max_levels -- both properties of the run.")
    print()
    print("  n     depth  per-level compressions           cumulative  "
          "secs  stop")
    depths, cums = [], []
    records = []
    for n in (64, 128, 256):
        r = run_stack(n)
        records.append((n, r))
        depths.append(float(r["depth"]))
        cums.append(r["cumulative"])
        shown = ", ".join("%.2fx" % x for x in r["comps"]) or "-"
        print("  %-5d %-6d %-32s %-11.1f %-5.0f %s"
              % (n, r["depth"], shown[:32], r["cumulative"],
                 r["elapsed"], (r["stop"] or "")[:40]))

    print()
    print("=== per-level structure, as observed ===")
    print("  Fan-out is REPORTED, not gated. A level below roughly the")
    print("  coarse fan-out collapses to one chunk, which was tempting")
    print("  to encode as a floor -- it is not encoded, because it is a")
    print("  property of these kernels and the refusal already catches")
    print("  the case it would have guarded.")
    print("  n     depth  level_n  chunks  compr    fanout  n/fanout")
    for n, r in records:
        for rec in r["levels"]:
            if rec["chunks"] is None:
                continue
            fan = rec["fanout"] or 0.0
            ratio = (rec["n"] / fan) if fan else 0.0
            print("  %-5d %-6d %-8d %-7d %-8.2f %-7.1f %.1f"
                  % (n, rec["depth"], rec["n"], rec["chunks"],
                     rec["compression"], fan, ratio))

    print()
    print("=== checks on the sweep ===")
    for name, vals in (("depth vs n", depths),
                       ("cumulative vs n", cums)):
        status, detail = wd.param_has_effect(vals)
        print("  [%s] %-24s %s" % (status, name, detail))
    status, detail = wd.monotone(cums, tol=0.5)
    print("  [%s] %-24s %s" % (status, "cumulative monotone", detail))

    # THE CHECK THAT WAS MISSING, and whose absence let a five- to
    # sevenfold overstatement survive several commits. Chunk count
    # cannot exceed the number of injection points, so at stride k it
    # is capped at n/k. With stride=8 every chunk count sat at that
    # cap and compression was read as emergent. param_has_effect
    # passed throughout, because the values did differ -- they just
    # differed because the harness parameter differed.
    measured, bounds = [], []
    for n, r in records:
        for rec in r["levels"]:
            if rec["chunks"] is None:
                continue
            measured.append(float(rec["chunks"]))
            bounds.append(float(len(range(0, rec["n"], 1))))
    status, detail = wd.not_harness_bound(measured, bounds)
    print("  [%s] %-24s %s" % (status, "not a harness artefact",
                               detail))
    print()
    print("  stop reasons are now properties of the run, so a depth")
    print("  difference across n is a real difference rather than an")
    print("  artefact of where a cutoff was placed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
