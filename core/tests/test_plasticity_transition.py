"""Directional tags and per-source scaling in Plasticity."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "py"))

from locus.plasticity import Plasticity  # noqa: E402


class TransitionTags(unittest.TestCase):

    def test_only_the_forward_direction_is_tagged(self):
        p = Plasticity()
        p.observe_transition([(1, 1.0)], [(2, 1.0)])
        self.assertGreater(p.trace(1, 2), 0.0)
        self.assertEqual(p.trace(2, 1), 0.0)

    def test_a_unit_may_follow_itself(self):
        p = Plasticity()
        p.observe_transition([(3, 1.0)], [(3, 1.0)])
        self.assertGreater(p.trace(3, 3), 0.0)

    def test_tags_decay_on_the_next_transition(self):
        p = Plasticity(trace_decay=0.5)
        p.observe_transition([(1, 1.0)], [(2, 1.0)])
        p.observe_transition([(7, 1.0)], [(8, 1.0)])
        self.assertAlmostEqual(p.trace(1, 2), 0.5)

    def test_transition_alone_changes_no_weight(self):
        p = Plasticity()
        p.observe_transition([(1, 1.0), (4, 1.0)], [(2, 1.0)])
        self.assertEqual(p.live_weights(), 0)
        self.assertEqual(p.consolidate(0.0), 0)
        self.assertEqual(p.live_weights(), 0)


class ScaleSources(unittest.TestCase):

    def learned(self):
        p = Plasticity(trace_decay=0.5, rate=1.0)
        p.observe_transition([(1, 1.0)], [(2, 1.0), (3, 3.0)])
        p.observe_transition([(9, 1.0)], [(5, 2.0)])
        p.consolidate(1.0)
        return p

    def test_row_sums_to_target_and_keeps_ratios(self):
        p = self.learned()
        before = p.weight(1, 3) / p.weight(1, 2)
        p.scale_sources(1.0, sources=[1])
        self.assertAlmostEqual(p.weight(1, 2) + p.weight(1, 3), 1.0)
        self.assertAlmostEqual(p.weight(1, 3) / p.weight(1, 2), before)

    def test_unlisted_sources_are_untouched(self):
        p = self.learned()
        w = p.weight(9, 5)
        self.assertEqual(p.scale_sources(1.0, sources=[1]), 1)
        self.assertEqual(p.weight(9, 5), w)

    def test_all_sources_when_none_listed(self):
        p = self.learned()
        self.assertEqual(p.scale_sources(1.0), 2)
        self.assertAlmostEqual(p.weight(9, 5), 1.0)

    def test_target_must_be_positive(self):
        with self.assertRaises(ValueError):
            Plasticity().scale_sources(0.0)


if __name__ == "__main__":
    unittest.main()
