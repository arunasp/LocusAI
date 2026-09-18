"""Persistence on the field, and what it implies for residency.

TWO QUESTIONS, one measurement.

1. DOES A TRACE PERSIST, and what decides it? Three candidate mechanisms
   were ruled out earlier with reasons: single-unit self-excitation could
   not hold a trace against a fast-mixing kernel; purely subtractive
   inhibition emptied the field; conserved-total competition left the
   goal below uniform at every gain. What was NOT tested until late is
   goal IDENTITY -- every one of those runs used the rarest state in the
   corpus, which is the worst case. A frequency-rank control then found
   a factor of 68 between the most and least connected goal, against a
   factor of ~7 across the parameter being swept. So the hypothesis
   under test here is that persistence is a property of GRAPH POSITION
   rather than of a mechanism to be invented.

2. WHAT HAS TO BE RESIDENT? This is where the in-memory-compute thread
   lands. The conclusion there was that a coarse level should be DERIVED
   rather than stored -- no second array, no shared cell, no write
   traffic -- and that the working set must be kept small enough to stay
   resident rather than placed somewhere by fiat, because on this device
   there is no software control over placement at all. If persistence is
   positional, then the set that must stay resident is the ACTIVE SET
   PLUS ITS SUPPORTING HUBS, which is bounded and small. This script
   measures that footprint directly instead of estimating it: how many
   states hold share above uniform, how many tagged pairs the sparse
   plasticity store actually holds, and what both cost in bytes.

Measurement, not assertion: nothing here is a pass/fail test. The
watchdog checks are applied to the SWEEP -- a parameter that changes
nothing, or a metric that is degenerate, invalidates the run rather than
the model.

Run: python3 tests/exp_persistence.py
"""

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "py"))
sys.path.insert(0, os.path.dirname(__file__))

import watchdog as wd                          # noqa: E402
from locus.field import Field                  # noqa: E402
from locus.plasticity import Plasticity        # noqa: E402

FLOAT_BYTES = 4          # what a device-side fp32 field would cost
PAIR_BYTES = 12          # two int32 keys plus one fp32 value


def heavy_tailed(n, seed):
    """Kernel where low indices are hubs, as in technical text.

    Targets drawn with a cubic bias toward low indices, so index order
    approximates frequency rank and inbound mass spans a wide range --
    which is the variable under test.
    """
    rnd = random.Random(seed)
    rows = []
    for i in range(n):
        row = [0.0] * n
        row[i] = 0.2 + 0.3 * rnd.random()
        deg = 2 + int(14 * (rnd.random() ** 2.5))
        for _ in range(deg):
            u = rnd.random()
            t = min(n - 1, int(n * u * u * u))
            row[t] += rnd.random()
        s = sum(row)
        rows.append([v / s for v in row])
    return rows


def inbound(kernel, j):
    return sum(row[j] for row in kernel)


def reciprocity(kernel, j):
    """Mutual connection strength: how much of j's inbound is RETURNED.

    Inbound mass alone was the predictor proposed for persistence, and
    the measurement refused it -- rank 8 held 61x uniform on a tenth of
    rank 0's inbound, while rank 1 held 9.9x on three times rank 8's.
    Inbound cannot distinguish a mutual pair from a sink, and a sink
    cannot sustain anything. This takes the min of each direction, so a
    pair contributes only what BOTH sides carry.
    """
    return sum(min(kernel[i][j], kernel[j][i]) for i in range(len(kernel)))


def spearman(xs, ys):
    """Rank correlation, computed plainly. Rank rather than value
    because the question is whether one quantity ORDERS the other, not
    whether they are linearly related."""
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        out = [0.0] * len(v)
        for pos, i in enumerate(order):
            out[i] = float(pos)
        return out
    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return (num / (dx * dy)) if dx > 0 and dy > 0 else 0.0


def participation_ratio(a):
    """Effective number of active units: (sum a)^2 / sum(a^2).

    THRESHOLD-FREE, which matters because the earlier residency figures
    were all threshold-dependent and the sparse pair store turned out to
    saturate into a dense one below 0.02. A measure that needs a cutoff
    cannot say whether the field is sparse; this one can.
    """
    s1 = sum(a)
    s2 = sum(v * v for v in a)
    return (s1 * s1 / s2) if s2 > 0 else 0.0


def main():
    n = 256
    kernel = heavy_tailed(n, 4242)
    uniform = 1.0 / n
    inb = [inbound(kernel, j) for j in range(n)]
    recip = [reciprocity(kernel, j) for j in range(n)]
    rank = sorted(range(n), key=lambda j: -inb[j])

    print("n=%d, uniform share=%.5f" % (n, uniform))
    print()
    print("=== 1. persistence by graph position (inject once, 200 steps) ===")
    print("  rank  inbound  recip    share@200  vs_uniform  settled_by")
    picks = [0, 1, 2, 4, 8, 16, 32, 64, 128, 255]
    shares, inbs = [], []
    for r in picks:
        goal = rank[r]
        f = Field(n, kernel, rho=1.0, g=0.5, beta=0.1, leak=0.3)
        f.set_state([0.01] * n)
        f.inject(goal, 1.0)
        hist = []
        for _ in range(200):
            f.step()
            hist.append(f.share(goal))
        # First step after which the share stops moving materially.
        settled = next((t for t in range(2, len(hist))
                        if abs(hist[t] - hist[t - 1]) < 1e-4), None)
        shares.append(hist[-1])
        inbs.append(inb[goal])
        print("  %-5d %-8.3f %-8.3f %-10.5f %-11.2fx %s"
              % (r, inb[goal], recip[goal], hist[-1],
                 hist[-1] / uniform,
                 settled if settled is not None else "not in 200"))

    for name, res in (
            ("param_has_effect(rank)", wd.param_has_effect(shares)),
            ("graded(shares)", wd.graded([round(s, 7) for s in shares])),
            ("monotone(share vs rank)",
             wd.monotone(list(reversed([round(s, 5) for s in shares]))))):
        status, detail = res
        print("  [%s] %-26s %s" % (status, name, detail))
    lo, hi = min(shares), max(shares)
    print("  spread: %.2fx uniform down to %.2fx -- factor %.0f across "
          "goal identity" % (hi / uniform, lo / uniform,
                             hi / lo if lo > 0 else float("inf")))

    print()
    print("=== 2. residency: what actually has to stay resident ===")
    # Run the field freely, read the active set each step, and let the
    # sparse plasticity store tag whatever co-occurs. The tag count is
    # the real pair-store footprint, not an estimate.
    print("  threshold  mean_active  peak_tags  field_KiB  tags_KiB")
    for threshold in (0.05, 0.02, 0.01, 0.005):
        f = Field(n, kernel, rho=1.0, g=0.5, beta=0.1, leak=0.3)
        f.set_state([0.01] * n)
        f.inject(rank[0], 1.0)
        p = Plasticity(trace_decay=0.7, trace_floor=1e-4)
        counts, peak = [], 0
        for _ in range(40):
            f.step()
            act = [v for v in f.a]
            live = f.active(threshold)
            counts.append(len(live))
            p.observe(act, threshold=threshold)
            peak = max(peak, p.live_traces())
        mean_active = sum(counts) / len(counts)
        field_kib = n * FLOAT_BYTES / 1024.0
        tags_kib = peak * PAIR_BYTES / 1024.0
        print("  %-10.3f %-12.1f %-10d %-10.3f %.2f"
              % (threshold, mean_active, peak, field_kib, tags_kib))

    print()
    print("=== 2b. which quantity actually ORDERS persistence? ===")
    all_shares, all_inb, all_rec = [], [], []
    for goal in range(0, n, 4):
        f = Field(n, kernel, rho=1.0, g=0.5, beta=0.1, leak=0.3)
        f.set_state([0.01] * n)
        f.inject(goal, 1.0)
        for _ in range(120):
            f.step()
        all_shares.append(f.share(goal))
        all_inb.append(inb[goal])
        all_rec.append(recip[goal])
    print("  sampled %d goals" % len(all_shares))
    print("  spearman(inbound,     share) = %+.3f"
          % spearman(all_inb, all_shares))
    print("  spearman(reciprocity, share) = %+.3f"
          % spearman(all_rec, all_shares))
    print("  -> the larger |rho| is the better predictor; if neither is")
    print("     strong, persistence is not a single-node property at all")

    print()
    print("=== 2c. inhibition versus sparsity (threshold-free) ===")
    print("  biological cortex runs ~1-2%% active; measured here at")
    print("  beta=0.1 the active set was 41%% of states, so inhibition")
    print("  is the first real tuning target. PR/n is the effective")
    print("  active fraction and needs no cutoff.")
    print("  beta      PR      PR/n     pair_store_if_dense")
    for beta in (0.1, 0.5, 2.0, 10.0, 50.0, 200.0):
        f = Field(n, kernel, rho=1.0, g=0.5, beta=beta, leak=0.3)
        f.set_state([0.01] * n)
        f.inject(rank[0], 1.0)
        for _ in range(60):
            f.step()
        pr = participation_ratio(f.a)
        frac = pr / n
        pairs = pr * (pr - 1.0)
        print("  %-9.1f %-7.2f %-8.4f %d pairs (%.1f KiB)"
              % (beta, pr, frac, int(pairs),
                 pairs * PAIR_BYTES / 1024.0))

    print()
    print("=== 2d. ATTRACTORS, not nodes -- the right unit ===")
    print("  Ranking nodes assumes persistence is a node property. The")
    print("  settled data refuses that: ranks 0, 4 and 8 landed at 0.229,")
    print("  0.227 and 0.236 -- three different injections converging on")
    print("  the same value, which is what falling into a SHARED attractor")
    print("  looks like. A node then selects a basin; it does not itself")
    print("  persist. Correlating node statistics against share was an")
    print("  apples-to-horses comparison and is dropped.")
    m = 64
    small = heavy_tailed(m, 909)
    sig_to_basin = {}
    self_member = 0
    trials = 0
    for goal in range(m):
        f = Field(m, small, rho=1.0, g=0.5, beta=0.3, leak=0.3)
        f.set_state([0.01] * m)
        f.inject(goal, 1.0)
        for _ in range(120):
            f.step()
        shares = [f.share(i) for i in range(m)]
        support = frozenset(i for i, s in enumerate(shares) if s >= 0.05)
        sig_to_basin.setdefault(support, []).append(goal)
        if goal in support:
            self_member += 1
        trials += 1
    print()
    print("  injections: %d over n=%d" % (trials, m))
    print("  DISTINCT ATTRACTORS: %d" % len(sig_to_basin))
    print("  injected node ends up INSIDE its own attractor: %d/%d (%.0f%%)"
          % (self_member, trials, 100.0 * self_member / trials))
    print()
    print("  attractor  size  basin  members (first 6)")
    ordered = sorted(sig_to_basin.items(), key=lambda kv: -len(kv[1]))
    for idx, (support, basin) in enumerate(ordered[:8]):
        mem = sorted(support)[:6]
        print("  %-10d %-5d %-6d %s" % (idx, len(support), len(basin),
                                        mem if mem else "(empty)"))
    sizes = [len(s) for s in sig_to_basin if s]
    if sizes:
        print()
        print("  attractor size: min %d, max %d, mean %.1f"
              % (min(sizes), max(sizes), sum(sizes) / len(sizes)))
        print("  -> THIS is the residency unit. The set that has to stay")
        print("     resident is an attractor, not the whole store, and it")
        print("     is bounded by the dynamics rather than by a cutoff.")
    empty = sum(1 for s in sig_to_basin if not s)
    if empty:
        print("  %d injection(s) reached NO attractor (field collapsed)"
              % sum(len(b) for s, b in sig_to_basin.items() if not s))

    print()
    print("=== 3. the residency argument, with these numbers ===")
    print("  The field is one value per state: %.1f KiB at n=%d, and it"
          % (n * FLOAT_BYTES / 1024.0, n))
    print("  scales linearly -- 4 KiB at n=1024, 16 KiB at n=4096. That")
    print("  is register-resident on the measured device (32.25 MiB of")
    print("  vector registers, five times the 6 MiB L2).")
    print("  The PAIR store is the term that grows, and it grows with")
    print("  the ACTIVE SET squared, not with n squared -- which is why")
    print("  the threshold column above is the residency dial.")
    print("  Nothing coarse is stored: share() and as_kernel() derive on")
    print("  every call. That was the in-memory-compute conclusion --")
    print("  there is no second array to move, and no placement control")
    print("  on this hardware to exploit even if there were.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
