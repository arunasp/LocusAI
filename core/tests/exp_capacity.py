"""Capacity: how many distinguishable states can the field hold?

WHY THIS IS THE BLOCKER. A task needs the system to distinguish many
situations. The first attractor count measured here was 11 over 64
injections at n=64, with ONE attractor capturing 48 of the 64 basins --
three quarters of all starting points converging on the same two-element
set. A system with about 11 distinguishable states cannot carry a coding
task, so capacity is what has to move before anything task-shaped is
worth attempting.

THE QUESTION THAT DECIDES IT: does capacity scale with n? If the count
stays near 11 however large the state space, the architecture is
finished at toy size and no amount of substrate helps. If it scales,
the ratio tells us what n a real task needs.

THREE LEVERS TESTED, and the third is the one worth knowing about:

  n      more states. The null hypothesis is that capacity is a
         property of the dynamics rather than of the space, in which
         case this does nothing.
  beta   inhibition. Sharper competition should mean more, smaller
         attractors -- but measured, beta saturates above 2.0 and does
         nothing further, so its range is narrow.
  kernel STRUCTURE. A modular kernel (dense within blocks, sparse
         between) should support one attractor per module, where a
         uniformly random kernel lets every basin drain into the same
         hubs. If this is the lever, capacity is a property of
         CONNECTIVITY and the design question moves from tuning to
         topology -- which would also match why hub states dominated
         every earlier measurement.

Measurement, not assertion. The watchdog checks guard the sweeps: a
parameter that changes nothing, or a metric pinned at a bound,
invalidates the run rather than the model.

Run: python3 tests/exp_capacity.py   (or `make capacity`)
"""

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "py"))
sys.path.insert(0, os.path.dirname(__file__))

import watchdog as wd                    # noqa: E402
from locus.cycle import Cycle            # noqa: E402


def random_kernel(n, seed, deg=6):
    """Uniformly random targets. No block structure, so nothing
    distinguishes one region from another."""
    rnd = random.Random(seed)
    rows = []
    for i in range(n):
        row = [0.0] * n
        row[i] = 0.3 + 0.2 * rnd.random()
        for _ in range(deg):
            row[rnd.randrange(n)] += rnd.random()
        s = sum(row)
        rows.append([v / s for v in row])
    return rows


def modular_kernel(n, seed, blocks=8, leak=0.05):
    """Dense within a block, sparse between.

    `leak` is the fraction of each row's mass that escapes its own
    block. Small but non-zero on purpose: a fully disconnected set of
    blocks would trivially give one attractor per block and prove
    nothing, and it would also be inadmissible for a trace chain, which
    needs excursions to return.
    """
    rnd = random.Random(seed)
    size = max(2, n // blocks)
    rows = []
    for i in range(n):
        row = [0.0] * n
        home = (i // size) * size
        for j in range(home, min(n, home + size)):
            row[j] = 0.5 + rnd.random()
        for _ in range(2):
            row[rnd.randrange(n)] += leak * (0.5 + rnd.random())
        s = sum(row)
        rows.append([v / s for v in row])
    return rows


def settle_support(c, goal, max_fast, hold=10, amount=0.2, tol=1e-5):
    """Run until BOTH criteria are met, reporting each separately.

    TWO DIFFERENT THINGS, and conflating them invalidated the first run
    of this experiment. `value_settled` is a fixed point: the state
    vector stops moving. `support_stable` is a stable ACTIVE SET: the
    same states stay above threshold for `hold` consecutive steps, even
    if their values keep changing.

    A field in a limit cycle never satisfies the first and can satisfy
    the second perfectly. Since the support is what identifies an
    attractor -- which states are active, not their exact values -- the
    second is the criterion capacity should use. The first run used only
    the first, reported `unsettled` equal to the trial count in every
    row, and I read the capacity numbers anyway.

    Returns (support, value_settled_at, support_stable_at), with None
    for a criterion never met within budget.
    """
    prev = list(c.field.a)
    last_support = None
    stable_for = 0
    value_at = None
    support_at = None
    for i in range(1, max_fast + 1):
        c.field.inject(goal, amount)
        c.field.step()
        delta = max(abs(a - b) for a, b in zip(c.field.a, prev))
        prev = list(c.field.a)
        if value_at is None and delta < tol:
            value_at = i
        support = frozenset(j for j, _ in c.decide())
        if support == last_support:
            stable_for += 1
            if support_at is None and stable_for >= hold:
                support_at = i
        else:
            stable_for = 0
            last_support = support
        if value_at is not None and support_at is not None:
            break
    return last_support or frozenset(), value_at, support_at


def attractors(n, kernel, beta, threshold=0.05, max_fast=400,
               stride=1):
    """Settle from every (or every `stride`-th) single-state injection
    and count distinct stable supports.

    The SUPPORT is the identity of an attractor -- which states are
    above threshold, not their values. Two injections landing on the
    same support are in the same basin.

    Reports value-convergence and support-stability separately, and a
    caller must check `no_support` is zero before reading any count:
    an unstable support means the number is a snapshot of a moving
    field, not an attractor count.
    """
    seen = {}
    no_value = 0
    no_support = 0
    value_steps = []
    support_steps = []
    for goal in range(0, n, stride):
        c = Cycle(n, kernel, threshold=threshold, beta=beta)
        c.reset_state()
        support, v_at, s_at = settle_support(c, goal, max_fast)
        if v_at is None:
            no_value += 1
        else:
            value_steps.append(v_at)
        if s_at is None:
            no_support += 1
        else:
            support_steps.append(s_at)
        seen.setdefault(support, []).append(goal)
    sizes = [len(s) for s in seen if s]
    basins = sorted((len(b) for b in seen.values()), reverse=True)
    total = sum(basins)
    return {
        "count": len(seen),
        "mean_size": (sum(sizes) / len(sizes)) if sizes else 0.0,
        "largest_basin": (basins[0] / total) if total else 0.0,
        "empty": sum(len(b) for s, b in seen.items() if not s),
        "no_value": no_value,
        "no_support": no_support,
        "mean_value_step": (sum(value_steps) / len(value_steps))
        if value_steps else None,
        "mean_support_step": (sum(support_steps) / len(support_steps))
        if support_steps else None,
        "trials": total,
    }


def main():
    print("=== 0. WHICH criterion is even reachable? (n=64, beta=0.3) ===")
    print("  budget  no_value  no_support  mean_value  mean_support")
    k64 = random_kernel(64, 71)
    reachable = None
    for budget in (80, 200, 400, 800):
        r = attractors(64, k64, beta=0.3, max_fast=budget, stride=8)
        print("  %-7d %-9d %-11d %-11s %s"
              % (budget, r["no_value"], r["no_support"],
                 "%.1f" % r["mean_value_step"]
                 if r["mean_value_step"] else "never",
                 "%.1f" % r["mean_support_step"]
                 if r["mean_support_step"] else "never"))
        if r["no_support"] == 0 and reachable is None:
            reachable = budget
    if reachable is None:
        print("  -> SUPPORT NEVER STABILISES at any budget tried. Every")
        print("     count below would be a snapshot of a moving field,")
        print("     so they are NOT reported as attractor counts.")
        return 1
    print("  -> support stabilises within %d steps; using that budget"
          % reachable)
    print("     (value convergence is a STRICTER criterion and a field")
    print("      in a limit cycle never meets it -- the support is what")
    print("      identifies an attractor, so that is the gate)")

    print()
    print("=== 1. does capacity scale with n? (random kernel) ===")
    print("  n    attractors  per_n    mean_size  largest  no_support")
    counts = []
    for n in (32, 48, 64, 96):
        r = attractors(n, random_kernel(n, 7 + n), beta=0.3,
                       max_fast=reachable)
        counts.append(r["count"])
        flag = "" if r["no_support"] == 0 else "  <-- UNSTABLE"
        print("  %-4d %-11d %-8.3f %-10.1f %-8.2f %d%s"
              % (n, r["count"], r["count"] / n, r["mean_size"],
                 r["largest_basin"], r["no_support"], flag))
    status, detail = wd.param_has_effect([float(c) for c in counts])
    print("  [%s] param_has_effect(n): %s" % (status, detail))
    status, detail = wd.monotone([float(c) for c in counts], tol=0.5)
    print("  [%s] monotone(count vs n): %s" % (status, detail))

    print()
    print("=== 2. does inhibition move it? (n=64, random kernel) ===")
    print("  beta   attractors  mean_size  largest  no_support")
    by_beta = []
    for beta in (0.1, 0.3, 1.0, 3.0):
        r = attractors(64, k64, beta=beta, max_fast=reachable)
        by_beta.append(r["count"])
        flag = "" if r["no_support"] == 0 else "  <-- UNSTABLE"
        print("  %-6.1f %-11d %-10.1f %-8.2f %d%s"
              % (beta, r["count"], r["mean_size"], r["largest_basin"],
                 r["no_support"], flag))
    status, detail = wd.param_has_effect([float(c) for c in by_beta])
    print("  [%s] param_has_effect(beta): %s" % (status, detail))
    print("  NOTE: `make operating-point` measures sparsity in the same")
    print("  configuration: the 1-2% active band is at beta >= 0.7.")

    print()
    print("=== 3. does kernel STRUCTURE move it? (n=64, beta=0.3) ===")
    print("  kernel            attractors  mean_size  largest  no_support")
    mod_counts = []
    r = attractors(64, k64, beta=0.3, max_fast=reachable)
    mod_counts.append(r["count"])
    print("  %-17s %-11d %-10.1f %-8.2f %d"
          % ("random", r["count"], r["mean_size"], r["largest_basin"],
             r["no_support"]))
    for blocks in (4, 8, 16):
        r = attractors(64, modular_kernel(64, 71, blocks=blocks),
                       beta=0.3, max_fast=reachable)
        mod_counts.append(r["count"])
        print("  %-17s %-11d %-10.1f %-8.2f %d"
              % ("modular x%d" % blocks, r["count"], r["mean_size"],
                 r["largest_basin"], r["no_support"]))
    status, detail = wd.param_has_effect([float(c) for c in mod_counts])
    print("  [%s] param_has_effect(structure): %s" % (status, detail))

    print()
    print("=== 4. what a task would need ===")
    print("  best measured capacity: %d distinguishable states"
          % max(max(counts), max(by_beta), max(mod_counts)))
    print("  a coding task must distinguish at least the files it")
    print("  touches, the errors it sees and the actions it can take --")
    print("  hundreds, not tens. Read the per_n column as the scaling")
    print("  claim and multiply; no single number here is sufficient.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
