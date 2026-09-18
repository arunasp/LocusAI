"""Property tests for eligibility traces and the three-factor rule.

The first test is load-bearing. Everything else checks ordinary
behaviour; ACTIVITY ALONE CHANGES NOTHING is the property that makes this
three-factor rather than Hebbian, and a drift back to writing weights
directly from coincidence would leave every other test here passing.

The rest map to specific biological claims, each written so it can fail:

  trace_is_local        a tag reads only its own synapse's pre and post.
                        Asserted by perturbing an unrelated unit and
                        requiring the tag to be unchanged -- the only way
                        to catch a global term leaking into a local rule.
  trace_decays          the tag has a lifetime, so a modulator that
                        arrives late finds less to act on.
  modulator_after_decay synaptic tagging and capture: product arriving
                        after the tag has gone is not captured.
  modulator_sign        one signal, two directions. No separate
                        depression rule.
  scaling_preserves_ratios
                        real synaptic scaling is multiplicative. A
                        global rescale that flattens relative sizes is
                        the measured failure this guards against.
  nothing_is_installed  weights start empty and only come from observed
                        activity. This is the property that cannot be
                        recovered once lost.
  bounded               repeated potentiation must not run away.
  sparse                the pair store must not grow without bound as
                        tags decay to nothing.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "py"))

from locus.plasticity import Plasticity  # noqa: E402


class ThreeFactorGate(unittest.TestCase):

    def test_activity_alone_changes_nothing(self):
        # THE test. Coincidence sets tags; only a modulator moves a
        # weight. If this fails the rule has become Hebbian.
        p = Plasticity()
        for _ in range(10):
            p.observe([1.0, 1.0, 0.0, 0.0])
        self.assertGreater(p.live_traces(), 0, "no tags were set at all")
        self.assertEqual(
            p.live_weights(), 0,
            "activity changed weights with no modulator -- this is "
            "Hebbian, not three-factor",
        )
        self.assertEqual(p.weight(0, 1), 0.0)

    def test_zero_modulator_is_a_no_op(self):
        p = Plasticity()
        p.observe([1.0, 1.0, 0.0])
        touched = p.consolidate(0.0)
        self.assertEqual(touched, 0)
        self.assertEqual(p.live_weights(), 0)

    def test_modulator_consolidates_tagged_pairs(self):
        p = Plasticity()
        p.observe([1.0, 1.0, 0.0])
        touched = p.consolidate(1.0)
        self.assertGreater(touched, 0, "modulator moved nothing")
        self.assertGreater(p.weight(0, 1), 0.0)

    def test_modulator_sign_flips_direction(self):
        up = Plasticity()
        up.observe([1.0, 1.0, 0.0])
        up.consolidate(1.0)
        down = Plasticity()
        down.observe([1.0, 1.0, 0.0])
        down.consolidate(-1.0)
        self.assertGreater(up.weight(0, 1), 0.0)
        self.assertLess(down.weight(0, 1), 0.0)
        self.assertAlmostEqual(up.weight(0, 1), -down.weight(0, 1),
                               places=12)


class Locality(unittest.TestCase):

    def test_trace_is_local(self):
        # A tag must depend only on its own pre and post. Changing an
        # unrelated unit's activity must not move it.
        quiet = Plasticity()
        quiet.observe([1.0, 1.0, 0.0, 0.0])
        loud = Plasticity()
        loud.observe([1.0, 1.0, 0.0, 9.0])
        self.assertAlmostEqual(
            quiet.trace(0, 1), loud.trace(0, 1), places=12,
            msg="an unrelated unit changed this synapse's tag -- a "
                "global term has leaked into a local rule",
        )

    def test_no_self_tag(self):
        p = Plasticity()
        p.observe([1.0, 1.0])
        self.assertEqual(p.trace(0, 0), 0.0, "a unit tagged itself")


class Timing(unittest.TestCase):

    def test_trace_decays(self):
        p = Plasticity(trace_decay=0.5)
        p.observe([1.0, 1.0])
        first = p.trace(0, 1)
        p.observe([0.0, 0.0])
        second = p.trace(0, 1)
        self.assertLess(second, first)
        self.assertAlmostEqual(second, first * 0.5, places=12)

    def test_modulator_after_decay_does_little(self):
        # Tagging and capture: product arriving after the tag has gone
        # finds nothing. Compare a prompt modulator with a late one.
        prompt = Plasticity(trace_decay=0.5)
        prompt.observe([1.0, 1.0])
        prompt.consolidate(1.0)

        late = Plasticity(trace_decay=0.5)
        late.observe([1.0, 1.0])
        for _ in range(12):
            late.observe([0.0, 0.0])
        late.consolidate(1.0)

        self.assertGreater(prompt.weight(0, 1), 0.0)
        self.assertLess(
            late.weight(0, 1), prompt.weight(0, 1) * 0.1,
            msg="a late modulator consolidated as much as a prompt one "
                "-- the tag has no effective lifetime",
        )

    def test_decayed_tags_are_dropped(self):
        # Otherwise the pair store grows without bound, holding
        # near-zero entries forever.
        p = Plasticity(trace_decay=0.5, trace_floor=1e-3)
        p.observe([1.0, 1.0, 1.0])
        self.assertGreater(p.live_traces(), 0)
        for _ in range(40):
            p.observe([0.0, 0.0, 0.0])
        self.assertEqual(p.live_traces(), 0,
                         "dead tags are still being kept")


class Homeostasis(unittest.TestCase):

    def test_scaling_preserves_ratios(self):
        # Multiplicative, so relative sizes survive. A rescale that
        # flattens them is the measured failure being guarded against.
        p = Plasticity(rate=1.0)
        p.observe([1.0, 2.0, 0.0])
        p.observe([1.0, 2.0, 0.0])
        p.consolidate(1.0)
        before = [p.weight(0, 1), p.weight(1, 0)]
        self.assertTrue(all(w != 0.0 for w in before))
        p.scale_to(1.0)
        after = [p.weight(0, 1), p.weight(1, 0)]
        self.assertAlmostEqual(
            before[0] / before[1], after[0] / after[1], places=9,
            msg="scaling changed the ratio of incoming weights",
        )

    def test_scaling_rejects_nonpositive_target(self):
        p = Plasticity()
        with self.assertRaises(ValueError):
            p.scale_to(0.0)

    def test_weights_are_bounded(self):
        p = Plasticity(rate=1.0, w_max=2.0)
        for _ in range(50):
            p.observe([1.0, 1.0])
            p.consolidate(1.0)
        self.assertLessEqual(abs(p.weight(0, 1)), 2.0,
                             "potentiation ran away past w_max")


class StreamsNotImports(unittest.TestCase):

    def test_nothing_is_installed(self):
        # No constructor path seeds weights or tags. This is the
        # property that cannot be recovered once lost.
        p = Plasticity()
        self.assertEqual(p.live_weights(), 0)
        self.assertEqual(p.live_traces(), 0)

    def test_as_kernel_invents_no_connectivity(self):
        # A row with no learned outgoing weight gets a self-loop, not a
        # uniform row. A uniform row would be connectivity the substrate
        # never formed.
        p = Plasticity(rate=1.0)
        p.observe([1.0, 1.0, 0.0, 0.0])
        p.consolidate(1.0)
        k = p.as_kernel(4)
        for r, row in enumerate(k):
            self.assertAlmostEqual(sum(row), 1.0, places=12,
                                   msg="row %d is not stochastic" % r)
        self.assertEqual(k[3][3], 1.0,
                         "an unlearned row was given invented outgoing "
                         "connectivity")

    def test_as_kernel_is_derived_not_stored(self):
        p = Plasticity(rate=1.0)
        p.observe([1.0, 1.0])
        p.consolidate(1.0)
        first = p.as_kernel(2)
        first[0][0] = 99.0
        second = p.as_kernel(2)
        self.assertNotEqual(second[0][0], 99.0,
                            "as_kernel handed out shared state")


class Construction(unittest.TestCase):

    def test_decay_must_be_a_proper_fraction(self):
        for bad in (0.0, 1.0, -0.1, 1.5):
            with self.assertRaises(ValueError):
                Plasticity(trace_decay=bad)


if __name__ == "__main__":
    unittest.main()
