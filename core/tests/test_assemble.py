"""Ingest: the system assembled from a values file, and bound by it.

The mechanisms were tested in isolation before this; what was missing
was a path that CONSTRUCTS them. These tests check the assembly binds
what it builds, that one audit surface covers both the dispatcher and
the graph, and that a values file cannot quietly permit more than it
says.
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "py"))

from locus.assemble import build, from_file, load_values   # noqa: E402
from locus.commit import Stage                             # noqa: E402

VALUES = {
    "costs": {"orient": 1.0, "delete-file": 0.05, "write-file": 0.25},
    "forbidden": ["self-modify"],
    "conflicts": {"delete-file": ["write-file"]},
    "classes": {"cleanup": "delete-file", "draft": "write-file"},
    "releasers": {
        "alarm": {"key": 1, "action": "orient", "class": "orient"},
        "wipe": {"key": 2, "action": "erase", "class": "self-modify"},
    },
}


def written(values):
    path = os.path.join(tempfile.mkdtemp(), "values.json")
    with open(path, "w") as fh:
        json.dump(values, fh)
    return path


class TestIngest(unittest.TestCase):

    def test_a_values_file_builds_the_whole_system(self):
        s = from_file(written(VALUES))
        self.assertEqual(s.constitution.costs["delete-file"], 0.05)
        self.assertEqual(set(s.constitution.forbidden), {"self-modify"})
        self.assertEqual(s.graph.conflicts["delete-file"],
                         frozenset({"write-file"}))
        self.assertTrue(s.dispatcher.fires("alarm"))

    def test_an_unknown_key_is_refused_not_ignored(self):
        # A typo'd "forbiden" would otherwise load as permitting
        # everything, and silently.
        bad = dict(VALUES)
        bad.pop("forbidden")
        bad["forbiden"] = ["self-modify"]
        with self.assertRaises(ValueError) as got:
            load_values(written(bad))
        self.assertIn("forbiden", str(got.exception))

    def test_the_file_decides_what_a_releaser_costs(self):
        s = build(VALUES)
        self.assertEqual(s.dispatcher._releasers["alarm"][2], 1.0)
        out = s.dispatcher.act("alarm", signals=4)
        self.assertEqual(out.stage, Stage.PREPARE)

    def test_a_forbidden_releaser_never_fires_through_the_system(self):
        s = build(VALUES)
        out = s.dispatcher.act("wipe", signals=99)
        self.assertTrue(out.vetoed)
        self.assertIsNone(out.action)

    def test_one_trace_covers_the_dispatcher_and_the_graph(self):
        s = build(VALUES)
        s.dispatcher.act("wipe")
        self.assertFalse(s.graph.link("cleanup", "draft"))
        sources = [r.source for r in s.trace.records]
        self.assertEqual(sources, ["veto", "link-refused"])
        self.assertEqual(len(s.trace.vetoes()), 2)

    def test_values_cannot_be_changed_after_construction(self):
        s = build(VALUES)
        with self.assertRaises(TypeError):
            s.constitution.costs["delete-file"] = 1.0
        with self.assertRaises(AttributeError):
            s.constitution.forbidden.add("anything")
        with self.assertRaises(TypeError):
            s.graph.conflicts["delete-file"] = frozenset()

    def test_a_priced_forbidden_act_is_refused_at_build(self):
        bad = dict(VALUES, costs=dict(VALUES["costs"], **{
            "self-modify": 1.0}))
        with self.assertRaises(ValueError):
            build(bad)

    def test_an_empty_file_builds_a_system_that_constrains_nothing(self):
        s = build({})
        self.assertEqual(dict(s.constitution.costs), {})
        self.assertEqual(len(s.trace), 0)
        self.assertTrue(s.graph.link("a", "b"))


if __name__ == "__main__":
    unittest.main()
