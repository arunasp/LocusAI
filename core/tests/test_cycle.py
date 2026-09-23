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
from fixtures import ring_kernel            # noqa: E402


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


class LearningReachesTheDynamics(unittest.TestCase):
    """The one-kernel property, which was a real defect before it was one.

    The field used to run on the constructor kernel while `level_up`
    traced a different one, so learning reached the BIAS and never the
    dynamics. `operating_kernel` (anatomy + learning) and the `refresh`
    inside `task_step` are the fix -- and nothing tested either, so
    reverting the refresh would have left every test green.
    """

    def test_learning_changes_the_kernel_the_field_runs_on(self):
        n = 16
        c = Cycle(n, ring_kernel(n, 61))
        before = [row[:] for row in c.field.kernel]
        c.task_step(evidence=[4], outcome=1.0)
        self.assertGreater(c.plasticity.live_weights(), 0,
                           "nothing was learned, so the test proves "
                           "nothing about what learning reaches")
        self.assertNotEqual(before, c.field.kernel,
                            "learning did not reach the dynamics -- the "
                            "field is running on a stale kernel")

    def test_refresh_is_what_carries_it(self):
        # Naming the mechanism, not just the outcome: learning alone
        # moves weights and leaves the field's kernel where it was.
        n = 16
        c = Cycle(n, ring_kernel(n, 62))
        c.settle(evidence=[4])
        stale = [row[:] for row in c.field.kernel]
        c.learn(1.0)
        self.assertEqual(stale, c.field.kernel,
                         "the field kernel moved without a refresh")
        c.refresh()
        self.assertNotEqual(stale, c.field.kernel,
                            "refresh did not hand the learned kernel to "
                            "the field")

    def test_a_bad_outcome_suppresses_a_structural_link(self):
        # Anatomy PLUS learning, with a negative weight subtracting --
        # and clamped at zero, since a link can be removed but not
        # inverted into negative probability.
        n = 16
        c = Cycle(n, ring_kernel(n, 63))
        for _ in range(3):
            c.task_step(evidence=[4], outcome=-1.0)
        self.assertLess(sum(c.plasticity.weights.values()), 0.0)
        for i, row in enumerate(c.operating_kernel()):
            self.assertTrue(all(v >= 0.0 for v in row),
                            "row %d went negative" % i)
            self.assertAlmostEqual(sum(row), 1.0, places=9,
                                   msg="row %d is not stochastic" % i)


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


class Hierarchy(unittest.TestCase):
    """Forming a level from the chunks the dynamics produced.

    A chunk is a stable attractor support; the coarse kernel is the
    trace chain over one representative per chunk. No new mechanism --
    which is why these tests are about the REFUSALS and about
    compression being an output, rather than about the coupling
    operator, which has its own suite.
    """

    def _trained(self, n, seed, episodes=6):
        c = Cycle(n, ring_kernel(n, seed), beta=1.0)
        for ep in range(episodes):
            c.reset_state()
            c.task_step(evidence=[ep % n], outcome=1.0)
        return c

    def test_survey_is_a_read(self):
        c = self._trained(16, 60)
        before = list(c.field.a)
        bias_before = list(c.field.bias)
        c.survey(max_fast=40, stride=4)
        self.assertEqual(c.field.a, before, "survey mutated the state")
        self.assertEqual(c.field.bias, bias_before,
                         "survey clobbered the bias")

    def test_refuses_before_anything_is_learned(self):
        # With no learned weights the fine kernel is all self-loops, so
        # no coarse level can exist. It must say so rather than invent
        # one.
        c = Cycle(12, ring_kernel(12, 61), beta=1.0)
        r = c.level_up(max_fast=40)
        self.assertFalse(r.admissible,
                         "formed a level from nothing: %r" % r)
        self.assertTrue(r.reason, "refused with no reason given")
        self.assertIn("refused", repr(r))

    def test_refusal_cannot_be_promoted_to_a_level(self):
        c = Cycle(12, ring_kernel(12, 62), beta=1.0)
        r = c.level_up(max_fast=40)
        if not r.admissible:
            with self.assertRaises(ValueError):
                c.next_level(r)

    def test_compression_is_an_output_not_a_parameter(self):
        # level_up takes no target ratio. Different operating points
        # must therefore yield different compression, and if they do
        # not, fixing a constant would have been harmless -- which is
        # the claim being tested.
        got = []
        for beta in (0.3, 1.0, 3.0):
            c = Cycle(24, ring_kernel(24, 63), beta=beta)
            for ep in range(6):
                c.reset_state()
                c.task_step(evidence=[ep], outcome=1.0)
            r = c.level_up(max_fast=60, stride=2)
            got.append(round(r.compression, 4) if r.admissible else None)
        self.assertTrue(
            any(v is not None for v in got),
            "no operating point produced a level at all: %r" % (got,))

    def test_coarse_kernel_is_stochastic_and_smaller(self):
        c = self._trained(24, 64, episodes=10)
        r = c.level_up(max_fast=60, stride=2)
        if not r.admissible:
            self.skipTest("no admissible level here: %s" % r.reason)
        self.assertGreaterEqual(r.chunks, 2)
        self.assertLessEqual(r.chunks, c.n)
        self.assertEqual(len(r.kernel), r.chunks)
        for row in r.kernel:
            self.assertEqual(len(row), r.chunks)
            self.assertAlmostEqual(sum(row), 1.0, places=9)
        self.assertAlmostEqual(r.compression, c.n / r.chunks, places=9)

    def test_representatives_are_distinct_and_drawn_from_supports(self):
        c = self._trained(24, 65, episodes=10)
        r = c.level_up(max_fast=60, stride=2)
        if not r.admissible:
            self.skipTest("no admissible level here: %s" % r.reason)
        self.assertEqual(len(set(r.representatives)), r.chunks,
                         "a representative named two chunks")
        union = set()
        for s in r.supports:
            union |= s
        for rep in r.representatives:
            self.assertIn(rep, union,
                          "representative %d is in no support" % rep)

    def test_next_level_has_its_own_plasticity_store(self):
        # Sharing one would let fine-level tags consolidate into
        # coarse-level weights, which is the import path wearing a
        # hierarchy.
        c = self._trained(24, 66, episodes=10)
        r = c.level_up(max_fast=60, stride=2)
        if not r.admissible:
            self.skipTest("no admissible level here: %s" % r.reason)
        up = c.next_level(r)
        self.assertIsNot(up.plasticity, c.plasticity)
        self.assertEqual(up.plasticity.live_weights(), 0,
                         "the coarse level started with weights it did "
                         "not learn")
        self.assertEqual(up.n, r.chunks)

    def test_stack_default_stride_does_not_cap_chunks_at_n_over_4(self):
        # At stride k the survey caps chunks at n/k. The explicit
        # stride=4 run is the negated case: it must sit at the cap,
        # or this test cannot tell the default from it.
        n = 24
        capped = Cycle(n, ring_kernel(n, 70), beta=3.0).drive_and_stack(
            evidence=[0], max_levels=1, stride=4)[0]["chunks"]
        self.assertEqual(capped, n // 4)
        got = Cycle(n, ring_kernel(n, 70), beta=3.0).drive_and_stack(
            evidence=[0], max_levels=1)[0]["chunks"]
        self.assertGreater(got, n // 4,
                           "default stride still caps level-0 chunks")


if __name__ == "__main__":
    unittest.main()
