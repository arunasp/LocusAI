"""Property tests for the simultaneous-update field.

The first test is the load-bearing one. Everything else here is ordinary
dynamics checking; SIMULTANEITY is the claim that distinguishes this from
the staged pipeline it replaces, and the only way to test it is to
permute the order units are computed in and require the result to be
bit-identical. An accidental in-place update -- writing into the same
list being read -- turns the field into a sequential sweep whose outcome
depends on index order, and it would otherwise look completely normal.

The rest exist because each corresponds to a defect already measured in
this project and reported as a result before being caught:

  no_dieoff          purely subtractive inhibition killed the whole field
                     in 11-19 steps at every beta > 0 tested. Shunting
                     must not be able to do that at any strength.
  gain_has_effect    global divisive normalisation erased the loop gain
                     so completely that three values of g produced
                     byte-identical output, voiding that whole sweep.
  graded_response    a constant offset plus a hard top-k gives a step,
                     and the step was mistaken for dynamics. Response to
                     injection size must be graded, not binary.
  injection_is_event re-applying a boost every step is a CLAMP, not
                     persistence; an earlier run measured a clamp and
                     called it a retained goal.
  active_is_a_read   the active set must not be a stage with side
                     effects, and must keep the sub-threshold values.
  share_not_logslope share is dimensionless and defined even when the
                     field level drifts; the log-slope it replaces never
                     settled and produced a retracted estimate.
"""

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "py"))

from locus.field import Field  # noqa: E402


def ring_kernel(n, seed, reach=3):
    """Row-stochastic kernel with local connectivity and a positive
    diagonal. Local rather than random because a random sparse kernel is
    frequently reducible, which makes failures say nothing about the
    dynamics under test."""
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


class Simultaneity(unittest.TestCase):

    def test_update_order_cannot_matter(self):
        # THE test. If any unit reads another unit's UPDATED value, the
        # result depends on the order of computation and the field is a
        # sequential sweep wearing a parallel name.
        n = 24
        k = ring_kernel(n, 1)
        base = [0.03 * (i + 1) for i in range(n)]
        orders = [list(range(n)), list(reversed(range(n)))]
        shuffled = list(range(n))
        random.Random(99).shuffle(shuffled)
        orders.append(shuffled)

        results = []
        for order in orders:
            f = Field(n, k)
            f.set_state(base)
            for _ in range(5):
                f.step(order=order)
            results.append(list(f.a))

        for other in results[1:]:
            for j in range(n):
                self.assertEqual(
                    results[0][j], other[j],
                    msg="state %d differs by update order: %r vs %r -- "
                        "the update is sequential, not simultaneous"
                        % (j, results[0][j], other[j]),
                )

    def test_step_does_not_alias_previous_state(self):
        # A returned state that IS the internal list would let a caller
        # corrupt the field, and would also mean the update wrote in
        # place.
        f = Field(6, ring_kernel(6, 2))
        f.inject(0, 1.0)
        out = f.step()
        out[0] = 999.0
        self.assertNotEqual(f.a[0], 999.0)


class Dynamics(unittest.TestCase):

    def test_no_dieoff_at_any_inhibition(self):
        # Subtractive inhibition emptied the field entirely. Shunting
        # divides the drive instead, so it must never be able to.
        n = 20
        k = ring_kernel(n, 3)
        for beta in (0.0, 0.05, 0.2, 1.0, 5.0):
            f = Field(n, k, beta=beta)
            f.set_state([0.05] * n)
            f.inject(0, 1.0)
            for _ in range(40):
                pass
            for _ in range(40):
                f.step()
            self.assertGreater(
                f.total(), 0.0,
                msg="field died at beta=%s -- inhibition is subtracting, "
                    "not shunting" % beta,
            )

    def test_gain_has_effect(self):
        # Three values of g once produced byte-identical output.
        n = 20
        k = ring_kernel(n, 4)
        finals = []
        for g in (0.2, 0.5, 0.9):
            f = Field(n, k, g=g)
            f.inject(0, 1.0)
            for _ in range(20):
                f.step()
            finals.append(round(f.total(), 9))
        self.assertEqual(len(set(finals)), 3,
                         "lateral gain is inert: %r" % (finals,))

    def test_self_excitation_has_effect(self):
        n = 20
        k = ring_kernel(n, 5)
        finals = []
        for rho in (0.5, 1.0, 1.5):
            f = Field(n, k, rho=rho)
            f.inject(0, 1.0)
            for _ in range(20):
                f.step()
            finals.append(round(f.share(0), 9))
        self.assertEqual(len(set(finals)), 3,
                         "self-excitation is inert: %r" % (finals,))

    def test_graded_response_to_injection(self):
        # Not a step function. A hard top-k produced one and it was
        # reported as a property of the dynamics.
        n = 20
        k = ring_kernel(n, 6)
        shares = []
        for amount in (0.05, 0.1, 0.2, 0.4, 0.8):
            f = Field(n, k)
            f.set_state([0.02] * n)
            f.inject(0, amount)
            for _ in range(10):
                f.step()
            shares.append(round(f.share(0), 6))
        self.assertGreater(len(set(shares)), 2,
                           "response is binary, not graded: %r" % (shares,))
        for a, b in zip(shares, shares[1:]):
            self.assertLessEqual(
                a, b + 1e-9,
                "more injection gave less share: %r" % (shares,))

    def test_leak_only_decays(self):
        # With no lateral drive and no self-excitation, leak must be the
        # only term and the state must fall monotonically.
        f = Field(8, ring_kernel(8, 7), rho=0.0, g=0.0, leak=0.5)
        f.inject(3, 1.0)
        prev = f.a[3]
        for _ in range(6):
            f.step()
            self.assertLess(f.a[3], prev + 1e-12)
            prev = f.a[3]
        self.assertLess(prev, 0.2, "leak did not decay the state")

    def test_injection_is_an_event_not_a_clamp(self):
        # Nothing re-applies an injection. Measuring a clamp and calling
        # it persistence is an error already made here.
        f = Field(12, ring_kernel(12, 8), rho=0.0, g=0.0, leak=0.9)
        f.inject(5, 1.0)
        start = f.a[5]
        for _ in range(10):
            f.step()
        self.assertLess(f.a[5], start * 0.5,
                        "state held without support -- something is "
                        "re-injecting")


class Reads(unittest.TestCase):

    def test_active_is_a_read(self):
        f = Field(10, ring_kernel(10, 9))
        f.inject(0, 1.0)
        f.step()
        before = list(f.a)
        steps_before = f.steps
        got = f.active(0.0)
        self.assertEqual(f.a, before, "active() mutated the field")
        self.assertEqual(f.steps, steps_before, "active() advanced time")
        self.assertTrue(all(isinstance(t, tuple) for t in got))

    def test_active_keeps_sub_threshold_values_available(self):
        # A hard top-k would discard these, and they are what an
        # eligibility trace and a deferral confidence read.
        n = 16
        f = Field(n, ring_kernel(n, 10))
        f.set_state([0.01 * (i + 1) for i in range(n)])
        f.step()
        high = f.active(0.05)
        self.assertLess(len(high), n, "threshold selected everything")
        self.assertEqual(len(f.a), n, "field lost states")
        self.assertTrue(any(v <= 0.05 for v in f.a),
                        "no sub-threshold values remain to read")

    def test_share_is_defined_and_dimensionless(self):
        n = 12
        f = Field(n, ring_kernel(n, 11))
        self.assertEqual(f.share(0), 0.0, "empty field should give 0")
        f.inject(0, 1.0)
        f.step()
        s = sum(f.share(i) for i in range(n))
        self.assertAlmostEqual(s, 1.0, places=9,
                               msg="shares do not sum to 1")

    def test_cycle_runs_fast_period_and_reports_history(self):
        n = 10
        f = Field(n, ring_kernel(n, 12))
        f.inject(0, 1.0)
        hist = f.cycle(fast=7)
        self.assertEqual(len(hist), 7, "cycle did not run 7 fast steps")
        self.assertEqual(f.steps, 7)
        self.assertEqual(f.cycles, 1)
        with self.assertRaises(ValueError):
            f.cycle(fast=0)


class Construction(unittest.TestCase):

    def test_kernel_shape_is_checked(self):
        with self.assertRaises(ValueError):
            Field(4, [[0.25] * 4] * 3)
        with self.assertRaises(ValueError):
            Field(4, [[0.25] * 3] * 4)

    def test_set_state_length_is_checked(self):
        f = Field(4, ring_kernel(4, 13))
        with self.assertRaises(ValueError):
            f.set_state([0.1, 0.2])


if __name__ == "__main__":
    unittest.main()
