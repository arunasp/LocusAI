"""Local error tags, tag capture, soft bounds and per-target scaling."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "py"))

from locus.plasticity import Plasticity  # noqa: E402


class ErrorTags(unittest.TestCase):

    def test_signed_tags_follow_the_error(self):
        p = Plasticity()
        p.observe_error([(1, 1.0)], [(2, 0.7), (3, -0.4)])
        self.assertAlmostEqual(p.trace(1, 2), 0.7)
        self.assertAlmostEqual(p.trace(1, 3), -0.4)

    def test_zero_error_sets_no_tag(self):
        p = Plasticity()
        p.observe_error([(1, 1.0)], [(2, 0.0)])
        self.assertEqual(p.live_traces(), 0)

    def test_error_alone_changes_no_weight(self):
        p = Plasticity()
        p.observe_error([(1, 1.0)], [(2, 1.0)])
        self.assertEqual(p.live_weights(), 0)


class Capture(unittest.TestCase):

    def test_consume_clears_the_tags(self):
        p = Plasticity()
        p.observe_error([(1, 1.0)], [(2, 1.0)])
        p.consolidate(1.0, consume=True)
        self.assertEqual(p.live_traces(), 0)
        w = p.weight(1, 2)
        p.consolidate(1.0, consume=True)
        self.assertEqual(p.weight(1, 2), w)

    def test_default_keeps_the_tags(self):
        p = Plasticity()
        p.observe_error([(1, 1.0)], [(2, 1.0)])
        p.consolidate(1.0)
        self.assertGreater(p.live_traces(), 0)


class SoftBounds(unittest.TestCase):

    def test_potentiation_saturates_below_the_bound(self):
        p = Plasticity(rate=0.5)
        last = 0.0
        for _ in range(50):
            p.observe_error([(1, 1.0)], [(2, 1.0)])
            p.consolidate(1.0, consume=True, bound=1.0)
            w = p.weight(1, 2)
            self.assertGreater(w, last)
            self.assertLessEqual(w, 1.0)
            last = w
        self.assertGreater(last, 0.99)

    def test_depression_from_zero_creates_no_weight(self):
        p = Plasticity()
        p.observe_error([(1, 1.0)], [(2, -1.0)])
        self.assertEqual(p.consolidate(1.0, consume=True, bound=1.0), 0)
        self.assertEqual(p.live_weights(), 0)

    def test_depression_never_goes_below_zero(self):
        p = Plasticity(rate=1.0)
        p.observe_error([(1, 1.0)], [(2, 1.0)])
        p.consolidate(1.0, consume=True, bound=1.0)
        for _ in range(20):
            p.observe_error([(1, 1.0)], [(2, -1.0)])
            p.consolidate(3.0, consume=True, bound=1.0)
            self.assertGreaterEqual(p.weight(1, 2), 0.0)


class LocalFactor(unittest.TestCase):

    def test_local_factor_scales_only_its_source(self):
        p = Plasticity(rate=0.5)
        p.observe_error([(1, 1.0), (4, 1.0)], [(2, 1.0)])
        p.consolidate(1.0, consume=True, local={1: 2.0})
        self.assertAlmostEqual(p.weight(1, 2), 1.0)
        self.assertAlmostEqual(p.weight(4, 2), 0.5)


class ScaleTargets(unittest.TestCase):

    def test_only_the_named_target_is_scaled(self):
        p = Plasticity(rate=1.0)
        p.observe_error([(1, 1.0), (4, 1.0)], [(2, 1.0), (3, 1.0)])
        p.consolidate(1.0, consume=True)
        w12, w42, w13 = p.weight(1, 2), p.weight(4, 2), p.weight(1, 3)
        self.assertEqual(p.scale_targets({2: 0.5}), 2)
        self.assertAlmostEqual(p.weight(1, 2), 0.5 * w12)
        self.assertAlmostEqual(p.weight(4, 2), 0.5 * w42)
        self.assertEqual(p.weight(1, 3), w13)


if __name__ == "__main__":
    unittest.main()
