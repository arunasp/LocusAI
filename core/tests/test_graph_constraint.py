"""Linking is constrained, and what cannot be prevented is recorded.

doc/core/CONSTITUTION.md calls the associative graph a re-description
engine: it finds routes between traces without intending to, and a route
is how a costed effect gets reached by an uncosted path. These test both
halves of the answer -- the direct link refused, and the indirect one
recorded, since no rule on single links can prevent it.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "py"))

from locus.decisions import DecisionTrace       # noqa: E402
from locus.graph import AssociativeGraph        # noqa: E402

CLASSES = {"gun": "weapon", "target": "person", "shed": "place",
           "rope": "tool"}
CONFLICTS = {"weapon": {"person"}}


def graph(trace=None, **kw):
    return AssociativeGraph(classes=CLASSES, conflicts=CONFLICTS,
                            trace=trace, **kw)


class TestRefusedLinks(unittest.TestCase):

    def test_incompatible_link_is_refused_not_raised(self):
        g = graph()
        self.assertFalse(g.link("gun", "target"))
        self.assertEqual(g.neighbours("gun"), {})
        self.assertEqual(g.neighbours("target"), {})

    def test_refusal_is_symmetric_whichever_way_declared(self):
        g = graph()
        self.assertFalse(g.link("target", "gun"))
        self.assertTrue(g.incompatible("target", "gun"))

    def test_permitted_links_are_unaffected(self):
        g = graph()
        self.assertTrue(g.link("gun", "shed", 0.9))
        self.assertTrue(g.link("shed", "target", 0.9))
        self.assertEqual(g.neighbours("gun"), {"shed": 0.9})

    def test_unclassified_keys_link_freely(self):
        g = graph()
        self.assertTrue(g.link("gun", "unknown-key"))

    def test_a_refusal_is_recorded(self):
        t = DecisionTrace()
        g = graph(trace=t)
        g.link("gun", "target")
        self.assertEqual(len(t.vetoes()), 1)
        self.assertEqual(t.records[0].source, "link-refused")
        self.assertEqual(t.records[0].action_class, ("weapon", "person"))

    def test_the_relation_cannot_be_widened_from_here(self):
        g = graph()
        with self.assertRaises(TypeError):
            g.conflicts["weapon"] = frozenset()
        with self.assertRaises(TypeError):
            g.classes["gun"] = "toy"


class TestIndirectReach(unittest.TestCase):

    def test_an_incompatible_pair_reached_over_two_hops_is_recorded(self):
        t = DecisionTrace()
        g = graph(trace=t, decay=0.9, floor=0.01, depth=3)
        # Each hop is permitted; only the ends are incompatible. No rule
        # on single links can stop this, so it must be visible instead.
        g.link("gun", "shed", 1.0)
        g.link("shed", "target", 1.0)
        g.spread(["gun"])
        noted = [r for r in t.records if r.source == "reached-indirectly"]
        self.assertEqual(len(noted), 1)
        self.assertEqual(set(noted[0].cue), {"gun", "target"})
        self.assertFalse(noted[0].vetoed)      # observed, not refused

    def test_nothing_is_recorded_when_no_pair_is_reached(self):
        t = DecisionTrace()
        g = graph(trace=t)
        g.link("shed", "rope", 1.0)
        g.spread(["shed"])
        self.assertEqual([r for r in t.records
                          if r.source == "reached-indirectly"], [])

    def test_spreading_still_returns_its_activation(self):
        g = graph(trace=DecisionTrace(), decay=0.9)
        g.link("gun", "shed", 1.0)
        got = g.spread(["gun"])
        self.assertGreater(got["shed"], 0.0)


if __name__ == "__main__":
    unittest.main()
