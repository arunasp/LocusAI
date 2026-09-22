"""Intrinsic excitability against hub capture.

  exp_homeostasis.py

On the random kernel the first capacity measurement found one attractor
capturing 48 of 64 basins (exp_capacity.py): hub states win whatever the
cue. Biology keeps a neuron from dominating by homeostatic intrinsic
plasticity (Desai, Rutherford & Turrigiano 1999): its excitability falls
when it fires more than its population and rises when it is silent,
without touching synapses.

Here, after every settle event, each unit's win frequency is updated as
a running mean over events (step 1/events, so no rate), and its
excitability becomes

    cue strength * (population mean frequency - own frequency)

bounded by the size of a cue. Epochs of one cue per state repeat until
the attractor map stops changing (an event, not a count; a harness cap
of 40 epochs is reported if reached). The control is the same field with
excitability left at zero.

Per condition: distinct attractors, the largest basin, cues settling in
an attractor that contains them, cues that never stabilised. One process
per (kernel, condition).
"""

import multiprocessing
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "py"))
sys.path.insert(0, HERE)

from exp_capacity import random_kernel  # noqa: E402
from locus.cycle import Cycle  # noqa: E402

N, BETA, MAX_FAST, HOLD, AMOUNT, CAP = 64, 0.3, 400, 10, 0.2, 40


def settle(c, goal):
    c.reset_state()
    last, held = None, 0
    for _ in range(MAX_FAST):
        c.field.inject(goal, AMOUNT)
        c.field.step()
        support = frozenset(i for i, _ in c.decide())
        if support == last:
            held += 1
            if held >= HOLD:
                return support
        else:
            held, last = 0, support
    return None


def summary(found, unstable):
    basins = sorted((len(v) for v in found.values()), reverse=True)
    own = sum(1 for s, goals in found.items() for g in goals if g in s)
    return {"count": len(found), "largest": basins[0] if basins else 0,
            "own": own, "unstable": unstable}


def run(spec):
    seed, homeo = spec
    t0 = time.process_time()
    c = Cycle(N, random_kernel(N, seed), beta=BETA)
    freq, events = [0.0] * N, 0
    history, prev, epoch = [], None, 0
    while True:
        found, unstable = {}, 0
        for goal in range(N):
            s = settle(c, goal)
            if s is None:
                unstable += 1
                continue
            if s:
                found.setdefault(s, []).append(goal)
            if homeo and s:
                events += 1
                for j in range(N):
                    freq[j] += ((1.0 if j in s else 0.0) - freq[j]) / events
                target = sum(freq) / N
                c.field.set_excitability(
                    [AMOUNT * (target - f) for f in freq])
        epoch += 1
        m = {k: sorted(v) for k, v in found.items()}
        history.append(summary(found, unstable))
        if not homeo or m == prev or epoch >= CAP:
            break
        prev = m
    return {"seed": seed, "homeo": homeo, "history": history,
            "capped": homeo and epoch >= CAP and m != prev,
            "cpu": time.process_time() - t0}


def main():
    specs = [(s, h) for s in (71, 72, 73) for h in (False, True)]
    t0 = time.time()
    with multiprocessing.get_context("fork").Pool(len(specs)) as pool:
        res = pool.map(run, specs, chunksize=1)
    print("random kernel n=%d, beta %.1f, one cue per state per epoch"
          % (N, BETA))
    print("%-5s %-12s %7s %11s %8s %9s %9s" % (
        "seed", "condition", "epochs", "attractors", "largest",
        "own-basin", "unstable"))
    for r in res:
        h = r["history"]
        last = h[-1]
        print("%-5d %-12s %7s %11d %8d %9d %9d%s" % (
            r["seed"], "homeostatic" if r["homeo"] else "control",
            len(h), last["count"], last["largest"], last["own"],
            last["unstable"], "  (harness cap)" if r["capped"] else ""))
        if r["homeo"]:
            print("      by epoch, attractors/largest: " + " ".join(
                "%d/%d" % (e["count"], e["largest"]) for e in h))
    print("resources: wall %.0f s, CPU %.0f core-s on %d processes" % (
        time.time() - t0, sum(r["cpu"] for r in res), len(res)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
