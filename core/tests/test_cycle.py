"""Property tests for the closed loop.

The loop contains no new mechanism -- Field, Plasticity and trace_chain
hold all of it. So these tests do not re-test dynamics or learning; they
test the WIRING, and specifically the four things the loop refuses to do,
each of which it refuses because a measurement said so:

  it does not read a decision before the field settles, because an
  injected state ends up inside its own settled attractum only 14% of
  the time -- reading early reads influences, not a decision;

  it does not set a bias directly, so attention cannot point where
  nothing was learned;

  it does not learn without an outcome, in either direction;

  it does not claim to have settled when it has not.

Plus the two properties that make it usable for a task at all: learning
survives an episode boundary, and a good outcome and a bad one move the
same weights in opposite directions.
"""

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "py"))
sys.path.insert(0, os.path.dirname(__file__))

import watchdog as wd  # noqa: E402
from locus.cycle import Cycle  # noqa: E402


def ring_kernel(n, seed, reach=3):
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


class Settling(unittest.TestCase):

    def test_settling_is_reported_not_assumed(self):
        # A caller that treats an unsettled field as a decision is
        # reading influences. So settle() reports, and a budget of one
        # fast step must come back False.
        n = 20
        c = Cycle(n, ring_kernel(n, 40))
        fast, settled = c.settle(evidence=[3], max_fast=1)
        self.assertEqual(fast, 1)
        self.assertFalse(settled,
                         "claimed to settle in a single fast step")

    def test_settling_converges_with_budget(self):
        n = 20
        c = Cycle(n, ring_kernel(n, 41))
        fast, settled = c.settle(evidence=[3], max_fast=200)
        self.assertTrue(settled, "did not settle in 200 fast steps")
        self.assertGreater(fast, 1, "settled suspiciously immediately")
        self.assertLess(fast, 200, "used the whole budget")

    def test_evidence_is_sustained_not_injected_once(self):
        # Evidence about a situation that persists must keep arriving.
        # If it were injected once it would decay and the settled state
        # would not reflect it.
        n = 20
        k = ring_kernel(n, 42)
        with_ev = Cycle(n, k)
        with_ev.settle(evidence=[7], max_fast=60)
        without = Cycle(n, k)
        without.settle(evidence=None, max_fast=60)
        self.assertGreater(
            with_ev.field.share(7), without.field.share(7),
            "sustained evidence left no trace in the settled state",
        )


class DecisionIsARead(unittest.TestCase):

    def test_decide_does_not_advance_or_mutate(self):
        n = 16
        c = Cycle(n, ring_kernel(n, 43))
        c.settle(evidence=[2], max_fast=40)
        before = list(c.field.a)
        steps_before = c.field.steps
        c.decide()
        c.decide()
        self.assertEqual(c.field.a, before, "decide() mutated the field")
        self.assertEqual(c.field.steps, steps_before,
                         "decide() advanced time")

    def test_decide_carries_values_not_just_indices(self):
        # A deferral threshold needs to know HOW strongly, which a hard
        # top-k would have discarded.
        n = 16
        c = Cycle(n, ring_kernel(n, 44))
        c.settle(evidence=[2], max_fast=40)
        got = c.decide()
        self.assertTrue(got, "nothing active after settling")
        for idx, val in got:
            self.assertIsInstance(idx, int)
            self.assertGreater(val, c.threshold)


class ThreeFactorGateInTheLoop(unittest.TestCase):

    def test_step_without_outcome_changes_no_weight(self):
        # A step with no outcome is still a legitimate step: it tags.
        n = 16
        c = Cycle(n, ring_kernel(n, 45))
        st = c.task_step(evidence=[4], outcome=0.0)
        self.assertGreater(st.tags, 0, "nothing was tagged at all")
        self.assertEqual(st.touched, 0,
                         "weights moved with no outcome -- the loop is "
                         "Hebbian, not three-factor")
        self.assertEqual(c.plasticity.live_weights(), 0)

    def test_outcome_moves_weights_and_sign_matters(self):
        n = 16
        k = ring_kernel(n, 46)
        good = Cycle(n, k)
        good.task_step(evidence=[4], outcome=1.0)
        bad = Cycle(n, k)
        bad.task_step(evidence=[4], outcome=-1.0)
        self.assertGreater(good.plasticity.live_weights(), 0)
        gw = sum(good.plasticity.weights.values())
        bw = sum(bad.plasticity.weights.values())
        self.assertGreater(gw, 0.0, "a good outcome did not potentiate")
        self.assertLess(bw, 0.0, "a bad outcome did not depress")

    def test_learning_reads_the_settled_state(self):
        # What gets tagged must be what the field DECIDED, not what it
        # was handed. Tagging before settling would bind the cue rather
        # than the outcome of the competition.
        n = 20
        c = Cycle(n, ring_kernel(n, 47))
        st = c.task_step(evidence=[6], outcome=1.0)
        active = set(i for i, _ in st.active)
        tagged = set()
        for (i, j) in c.plasticity.weights:
            tagged.add(i)
            tagged.add(j)
        self.assertTrue(
            tagged.issubset(active),
            "weights were formed between states that were not active in "
            "the settled set: %s outside %s"
            % (sorted(tagged - active), sorted(active)),
        )


class OrientIsDerived(unittest.TestCase):

    def test_bias_is_zero_before_anything_is_learned(self):
        n = 16
        c = Cycle(n, ring_kernel(n, 48))
        c.settle(evidence=[3], max_fast=30)
        bias = c.orient()
        self.assertEqual(
            max(abs(v) for v in bias), 0.0,
            "a bias appeared before any outcome was received",
        )

    def test_bias_points_only_where_weights_exist(self):
        n = 20
        c = Cycle(n, ring_kernel(n, 49))
        c.task_step(evidence=[8], outcome=1.0)
        bias = c.orient()
        self.assertGreater(max(bias), 0.0, "nothing was learned")
        targets = set(j for (_, j) in c.plasticity.weights)
        for j, v in enumerate(bias):
            if v != 0.0:
                self.assertIn(
                    j, targets,
                    "bias %.4f at state %d with no learned weight into "
                    "it" % (v, j),
                )

    def test_bias_is_bounded_by_strength(self):
        n = 20
        c = Cycle(n, ring_kernel(n, 50), bias_strength=0.2)
        for _ in range(8):
            c.task_step(evidence=[8], outcome=1.0)
        bias = c.field.bias
        self.assertLessEqual(
            max(abs(v) for v in bias), 0.2 + 1e-12,
            "repeated potentiation pushed the bias past its strength",
        )


class UsableForATask(unittest.TestCase):

    def test_learning_survives_an_episode_boundary(self):
        # The point of learning is that it outlives the episode that
        # produced it. reset_state clears activation only.
        n = 20
        c = Cycle(n, ring_kernel(n, 51))
        c.task_step(evidence=[5], outcome=1.0)
        learned = dict(c.plasticity.weights)
        self.assertTrue(learned, "nothing learned in the first episode")
        c.reset_state()
        self.assertEqual(c.plasticity.weights, learned,
                         "resetting activation destroyed learning")
        self.assertLess(max(c.field.a), 0.05,
                        "reset did not clear activation")

    def test_repeated_reward_accumulates(self):
        # Across episodes, the same evidence and outcome must build.
        n = 20
        c = Cycle(n, ring_kernel(n, 52))
        totals = []
        for _ in range(5):
            c.reset_state()
            c.task_step(evidence=[9], outcome=1.0)
            totals.append(sum(c.plasticity.weights.values()))
        status, detail = wd.in_responsive_range(totals, lo=0.0)
        self.assertEqual(status, "pass",
                         "weight totals are pinned, so the sweep "
                         "measures a bound: %s" % detail)
        for a, b in zip(totals, totals[1:]):
            self.assertGreaterEqual(
                b, a - 1e-12,
                "repeated reward did not accumulate: %r" % (totals,))

    def test_step_reports_what_it_did(self):
        n = 16
        c = Cycle(n, ring_kernel(n, 53))
        st = c.task_step(evidence=[1], outcome=0.5)
        self.assertEqual(st.modulator, 0.5)
        self.assertGreaterEqual(st.fast_steps, 1)
        self.assertIsInstance(st.settled, bool)
        self.assertIn("Step(", repr(st))


if __name__ == "__main__":
    unittest.main()
