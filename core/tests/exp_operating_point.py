"""The operating point: one parameter, three criteria, measured together.

WHY THIS EXISTS. Inhibition has been measured three times against three
different requirements, each in its own configuration, and they
disagree:

  SPARSITY   cortex runs 1-2% active. Participation ratio put that band
             between beta 0.1 and 0.5, above which inhibition saturates
             and does nothing further.
  CAPACITY   how many distinguishable states the field holds. At
             beta=0.1 there are 2 attractors with one basin holding
             98%; at beta>=1.0 it reaches 64 of 64 at n=64. Capacity
             wants HIGH inhibition.
  COMPRESSION n / chunk count, which decides whether a hierarchy exists
             at all. Capacity near n means every state is its own
             attractor -- maximum distinguishability and ZERO
             compression, so no level above it is possible. Compression
             wants LOW inhibition.

Capacity and compression are the same dial pulled in opposite
directions, and a task needs both: enough states to tell situations
apart, and enough compression that levels can stack. Three criteria
measured in three separate runs cannot be traded off, because nothing
guarantees the configurations were comparable. THIS measures all three
from ONE configuration per beta, which is the whole point.

WHAT WOULD FALSIFY THE DESIGN: no beta where all three are acceptable
at once. That would mean the single inhibition parameter cannot serve
the three jobs and the model needs a second control -- per-area
competition is the obvious candidate, and it is already listed as
missing in ARCHITECTURE.

Run: python3 tests/exp_operating_point.py   (or `make operating-point`)
"""

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "py"))
sys.path.insert(0, os.path.dirname(__file__))

import watchdog as wd                    # noqa: E402
from locus.cycle import Cycle            # noqa: E402


def ring_kernel(n, seed, reach=3):
    """Local connectivity with a positive diagonal. Local rather than
    random because a random sparse kernel is frequently reducible, and
    then a failure says nothing about the parameter under test."""
    rnd = random.Random(seed)
    rows = []
    for i in range(n):
        row = [0.0] * n
        row[i] = 0.4 + 0.3 * rnd.random()
        for d in range(1, reach + 1):
            w = (0.6 ** d) * (0.4 + rnd.random())
            row[(i + d) % n] += w
            row[(i - d) % n] += w
        s = sum(row)
        rows.append([v / s for v in row])
    return rows


def participation_ratio(a):
    """Effective number of active units: (sum a)^2 / sum(a^2).

    Threshold-free, which matters: an earlier reading of 41% active was
    an artefact of a chosen cutoff, and this measure needs none.
    """
    s1 = sum(a)
    s2 = sum(v * v for v in a)
    return (s1 * s1 / s2) if s2 > 0 else 0.0


def evaluate(n, kernel, beta, episodes=10, stride=4, max_fast=120):
    """One configuration, all three criteria.

    Same Cycle, same learning, same settling for every measurement, so
    the three numbers are actually comparable -- which they were not
    when each was measured in its own run.
    """
    c = Cycle(n, kernel, beta=beta)
    for ep in range(episodes):
        c.reset_state()
        c.task_step(evidence=[(ep * 7) % n], outcome=1.0)

    # Sparsity, from a settled field.
    c.reset_state()
    for _ in range(max_fast):
        c.field.inject(0, 0.2)
        c.field.step()
    pr = participation_ratio(c.field.a)

    supports, unstable = c.survey(max_fast=max_fast, stride=stride)
    level = c.level_up(max_fast=max_fast, stride=stride)
    return {
        "pr": pr,
        "pr_frac": pr / n,
        "attractors": len(supports),
        "unstable": unstable,
        "chunks": level.chunks if level.admissible else None,
        "compression": level.compression if level.admissible else None,
        "reason": level.reason,
        "weights": c.plasticity.live_weights(),
    }


def main():
    n = 64
    kernel = ring_kernel(n, 808)
    betas = (0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0)

    print("n=%d, %d episodes per configuration" % (n, 10))
    print()
    print("  beta   PR     PR/n     attractors  unstable  chunks  compr")
    rows = []
    for beta in betas:
        r = evaluate(n, kernel, beta)
        rows.append((beta, r))
        print("  %-6.1f %-6.2f %-8.4f %-11d %-9d %-7s %s"
              % (beta, r["pr"], r["pr_frac"], r["attractors"],
                 r["unstable"],
                 r["chunks"] if r["chunks"] is not None else "-",
                 ("%.2fx" % r["compression"])
                 if r["compression"] is not None else "-"))

    print()
    print("=== do the three criteria respond to beta at all? ===")
    for name, key in (("sparsity (PR/n)", "pr_frac"),
                      ("capacity (attractors)", "attractors"),
                      ("compression", "compression")):
        vals = [float(r[key]) for _, r in rows if r[key] is not None]
        status, detail = wd.param_has_effect(vals)
        print("  [%s] %-24s %s" % (status, name, detail))

    print()
    print("=== is there a JOINT operating point? ===")
    print("  requires: PR/n <= 0.05 (near the 1-2%% biological band),")
    print("            attractors >= n/4 (enough to tell states apart),")
    print("            compression >= 1.5x (a level above is possible)")
    joint = []
    for beta, r in rows:
        ok = (r["pr_frac"] <= 0.05
              and r["attractors"] >= n / 4
              and r["compression"] is not None
              and r["compression"] >= 1.5)
        if ok:
            joint.append(beta)
    if joint:
        print("  FOUND: beta in %s satisfies all three" % (joint,))
    else:
        print("  NONE. No single beta satisfies all three at once, so")
        print("  one inhibition parameter cannot serve the three jobs.")
        print("  That is a falsification of the single-control design,")
        print("  not a tuning failure -- per-area competition is the")
        print("  candidate second control, already listed as missing in")
        print("  doc/core/ARCHITECTURE.md.")
        print()
        print("  closest by criterion:")
        best_sp = min(rows, key=lambda t: t[1]["pr_frac"])
        best_cap = max(rows, key=lambda t: t[1]["attractors"])
        comp = [t for t in rows if t[1]["compression"] is not None]
        print("    sparsest    beta=%.1f at PR/n %.4f"
              % (best_sp[0], best_sp[1]["pr_frac"]))
        print("    most states beta=%.1f at %d attractors"
              % (best_cap[0], best_cap[1]["attractors"]))
        if comp:
            best_c = max(comp, key=lambda t: t[1]["compression"])
            print("    most compr. beta=%.1f at %.2fx"
                  % (best_c[0], best_c[1]["compression"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
