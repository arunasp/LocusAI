"""The constitution wired into dispatch: cost, veto, and the audit.

doc/core/CONSTITUTION.md's "Not implemented" list said nothing produced
reversibility and nothing recorded decisions. These test what that
wiring must do, not that it merely exists.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "py"))

from locus.commit import Cascade, Stage          # noqa: E402
from locus.constitution import Constitution      # noqa: E402
from locus.constitution import UNCLASSIFIED_REVERSIBILITY  # noqa: E402
from locus.decisions import DecisionTrace        # noqa: E402
from locus.pathways import Dispatcher            # noqa: E402
from locus.procedural import Consolidator        # noqa: E402
from locus.store import Store                    # noqa: E402


def dispatcher(constitution=None):
    store = Store()
    return Dispatcher(store, Consolidator(store), cascade=Cascade(),
                      constitution=constitution)


class TestCostComesFromTheClass(unittest.TestCase):

    def test_declared_class_sets_reversibility(self):
        c = Constitution(costs={"reversible": 1.0, "grave": 0.1})
        d = dispatcher(c)
        d.add_releaser("a", 1, "act-a", action_class="reversible")
        d.add_releaser("b", 2, "act-b", action_class="grave")
        self.assertEqual(d._releasers["a"][2], 1.0)
        self.assertEqual(d._releasers["b"][2], 0.1)

    def test_unnamed_class_is_the_most_expensive_not_the_least(self):
        d = dispatcher(Constitution(costs={"known": 1.0}))
        d.add_releaser("x", 1, "act-x", action_class="never-declared")
        self.assertEqual(d._releasers["x"][2], UNCLASSIFIED_REVERSIBILITY)
        self.assertLess(UNCLASSIFIED_REVERSIBILITY, 1.0)

    def test_a_grave_act_does_not_commit_on_releaser_evidence(self):
        c = Constitution(costs={"trivial": 1.0, "grave": 0.1})
        d = dispatcher(c)
        d.add_releaser("t", 1, "act-t", action_class="trivial")
        d.add_releaser("g", 2, "act-g", action_class="grave")
        self.assertEqual(d.act("t", signals=4).stage, Stage.PREPARE)
        self.assertLess(d.act("g", signals=4).stage, Stage.COMMIT)


class TestVeto(unittest.TestCase):

    def test_forbidden_act_never_runs_however_many_signals(self):
        c = Constitution(costs={"fine": 1.0}, forbidden={"never"})
        d = dispatcher(c)
        d.add_releaser("n", 1, "act-n", action_class="never")
        out = d.act("n", signals=999)
        self.assertTrue(out.vetoed)
        self.assertIsNone(out.action)
        self.assertEqual(out.stage, Stage.HOLD)
        self.assertEqual(out.source, "none")

    def test_veto_is_recorded_and_teaches_nothing(self):
        c = Constitution(forbidden={"never"})
        d = dispatcher(c)
        d.add_releaser("n", 1, "act-n", action_class="never")
        before = d.consolidator.cost(1)
        for _ in range(5):
            d.act("n")
        # Recorded five times...
        self.assertEqual(len(d.trace.vetoes()), 5)
        self.assertEqual(len(d.trace.records), 5)
        # ...and none of it reached the learning channel: repeated
        # refusal must not make the refused route cheaper, or the system
        # would be learning to approach the veto.
        self.assertEqual(d.consolidator.cost(1), before)

    def test_a_cost_cannot_buy_a_forbidden_act(self):
        with self.assertRaises(ValueError):
            Constitution(costs={"never": 1.0}, forbidden={"never"})


class TestAudit(unittest.TestCase):

    def test_every_decision_is_recorded_with_its_cost(self):
        c = Constitution(costs={"grave": 0.1}, forbidden={"never"})
        d = dispatcher(c)
        d.add_releaser("g", 1, "act-g", action_class="grave")
        d.add_releaser("n", 2, "act-n", action_class="never")
        d.act("g")
        d.act("n")
        rec = d.trace.records
        self.assertEqual(len(rec), 2)
        self.assertEqual(rec[0].action_class, "grave")
        self.assertEqual(rec[0].reversibility, 0.1)
        self.assertTrue(rec[1].vetoed)
        self.assertEqual(d.trace.rate(), 0.5)

    def test_the_audited_party_cannot_edit_the_record(self):
        t = DecisionTrace()
        t.record(cue="c", action_class=None, reversibility=1.0,
                 evidence=0.6, signals=1, stage=int(Stage.COMMIT))
        got = t.records
        self.assertIsInstance(got, tuple)
        with self.assertRaises(AttributeError):
            got[0].stage = 0                      # frozen record

    def test_render_is_for_a_person(self):
        t = DecisionTrace()
        t.record(cue="c", action_class="grave", reversibility=0.1,
                 evidence=0.6, signals=2, stage=int(Stage.PREPARE),
                 vetoed=True, source="veto")
        line = t.render()[0]
        self.assertIn("grave", line)
        self.assertIn("VETOED", line)


class TestWithoutAConstitution(unittest.TestCase):

    def test_dispatch_is_unchanged_when_none_is_supplied(self):
        d = dispatcher(None)
        d.add_releaser("a", 1, "act-a")
        out = d.act("a", signals=4)
        self.assertEqual(out.source, "instinct")
        self.assertFalse(out.vetoed)
        self.assertEqual(out.stage, Stage.PREPARE)


if __name__ == "__main__":
    unittest.main()
