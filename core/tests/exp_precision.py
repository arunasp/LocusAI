"""Kernel-weight precision: which attractor does a cue reach at FP8/FP4?

Kernel reads dominate a field update (O(n^2) against O(n) for the state),
so storing the kernel at lower precision is the bandwidth lever. This
asks whether it changes the dynamics.

FP8 is E4M3 per weight. FP4 is E2M1 with one E4M3 scale per 16 weights,
the block format DeepSeek-V4.1-Flash uses for its main KV cache.

Each cue is settled from rest: the support must hold for `HOLD` steps. A
settle that runs out of budget is reported as capped, and any capped
settle makes the run invalid rather than a result.

Run: python3 tests/exp_precision.py   (or `make precision`)
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "py"))
sys.path.insert(0, os.path.dirname(__file__))

from locus.field import Field            # noqa: E402
from test_cycle import ring_kernel       # noqa: E402
from exp_capacity import random_kernel   # noqa: E402

THRESHOLD = 0.05
HOLD = 10
BUDGET = 200
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
BLOCK = 16


def e4m3(x):
    """Round a non-negative value to FP8 E4M3 (3 mantissa bits, max 448)."""
    if x <= 0.0:
        return 0.0
    e = max(math.floor(math.log2(x)), -6)
    m = round(x / 2 ** e * 8) / 8
    return min(448.0, m * 2 ** e)


def quantize(kernel, fmt):
    out = []
    for row in kernel:
        if fmt == "fp8":
            out.append([e4m3(v) for v in row])
            continue
        q = []
        for b in range(0, len(row), BLOCK):
            chunk = row[b:b + BLOCK]
            s = e4m3(max(chunk) / E2M1[-1])
            if s == 0.0:
                q += [0.0] * len(chunk)
                continue
            q += [min(E2M1, key=lambda c: abs(c - v / s)) * s
                  for v in chunk]
        out.append(q)
    return out


def supports(kernel, n, beta):
    """Settled support per cue; None where the budget ran out."""
    res = []
    for cue in range(n):
        f = Field(n, kernel, beta=beta)
        f.set_state([0.02] * n)
        last, held = None, 0
        for _ in range(BUDGET):
            f.inject(cue, 0.2)
            f.step()
            s = frozenset(i for i, v in enumerate(f.a) if v > THRESHOLD)
            held = held + 1 if s == last else 0
            last = s
            if held >= HOLD:
                break
        res.append(last if held >= HOLD else None)
    return res


def main():
    capped_total = 0
    print("kernel n  beta repertoire  fp8_same fp4_same  (cues keeping "
          "their basin)")
    for name, make in (("ring", ring_kernel), ("random", random_kernel)):
        for n in (32, 64):
            kernel = make(n, 7)
            for beta in (0.3, 1.0, 3.0):
                ref = supports(kernel, n, beta)
                row = []
                capped = sum(r is None for r in ref)
                for fmt in ("fp8", "fp4"):
                    q = supports(quantize(kernel, fmt), n, beta)
                    capped += sum(r is None for r in q)
                    row.append(sum(a == b for a, b in zip(ref, q)))
                capped_total += capped
                print("%-6s %-2d %-4.1f %-10d %2d/%-2d    %2d/%-2d  capped=%d"
                      % (name, n, beta, len(set(ref)), row[0], n, row[1], n,
                         capped))
    if capped_total:
        print("INVALID: %d settle(s) ended on the budget" % capped_total)
        return 2
    print("DONE: every settle ended on its own event")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
