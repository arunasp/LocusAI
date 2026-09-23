"""Behavioural tests for the LocusAI pathway layer."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "py"))

from locus import (  # noqa: E402
    AssociativeGraph, Cascade, Consolidator, Constitution, Dispatcher,
    NoveltyGate, Pathway, Restabilize, Stage, Store, Tier,
)


def make_recurring(store, key):
    """Give a trace a winner set that recurs.

    Promotion needs a recurring WINNER SET, not repetition alone
    (doc/core/ROADMAP.md stage 1), so a trace being reinforced has to
    have won a moment more than once. One trace active across two ticks
    is the smallest such set.
    """
    for _ in range(2):
        store.excite(key, 3.0)
        store.tick()


class StoreTest(unittest.TestCase):
    def setUp(self):
        self.store = Store()
        self.addCleanup(self.store.close)

    def test_lookup_does_not_destabilise(self):
        self.store.put(1, "abc")
        with self.store.lookup(1) as lease:
            self.assertEqual(lease.data, b"abc")
            self.assertFalse(lease.labile)

    def test_retrieve_destabilises_and_rewrites(self):
        self.store.put(1, "old")
        with self.store.retrieve(1) as lease:
            self.assertTrue(lease.labile)
        self.assertIs(self.store.restabilize(1, "new"),
                      Restabilize.WRITTEN)
        with self.store.lookup(1) as lease:
            self.assertEqual(lease.data, b"new")

    def test_unrestabilised_trace_is_lost(self):
        self.store.put(1, "x")
        self.store.retrieve(1).release()
        for _ in range(3):
            self.store.tick()
        self.assertEqual(self.store.stats()["labile_lost"], 1)

    def test_active_set_bounded_by_competition(self):
        store = Store(active_k=3, kwta_temp=0.0)
        self.addCleanup(store.close)
        for key in range(1, 9):
            store.put(key, "y", salience=2.0 + key)
        store.tick()
        self.assertEqual(store.stats()["active_traces"], 3)

    def test_miss_returns_none(self):
        self.assertIsNone(self.store.lookup(404))

    def test_kwta_primitive(self):
        winners = self.store.kwta([0.1, 0.9, 0.4, 0.8], 2)
        self.assertEqual(winners, [False, True, False, True])


class GraphTest(unittest.TestCase):
    def setUp(self):
        self.graph = AssociativeGraph(decay=0.5, floor=0.01)
        self.graph.link("a", "b", 1.0)
        self.graph.link("b", "c", 1.0)
        self.graph.link("c", "d", 1.0)

    def test_activation_decays_with_distance(self):
        act = self.graph.spread(["a"], depth=3)
        self.assertGreater(act["b"], act["c"])
        self.assertGreater(act["c"], act["d"])

    def test_depth_bounds_reach(self):
        self.assertNotIn("d", self.graph.spread(["a"], depth=2))

    def test_prefetch_excludes_origin(self):
        warm = self.graph.prefetch_set(["a"])
        self.assertNotIn("a", warm)
        self.assertEqual(warm[0], "b")

    def test_weight_saturates(self):
        self.graph.link("a", "b", 1.0)
        self.assertEqual(self.graph.neighbours("a")["b"], 1.0)


class NoveltyForgettingTest(unittest.TestCase):
    """A gate in front of a finite memory must forget what the memory
    forgot, or an event can never become novel again.

    This replaces NoveltyGate.forget(), which was deleted as inert: the
    capability was real but nothing ever called it, and an uncalled
    deletion path invites someone to trust it. Pruning now rides the
    novelty comparison itself, so it cannot fall out of use.
    """

    def test_an_evicted_trace_stops_counting_as_known(self):
        with Store(capacity=2) as store:
            gate = NoveltyGate(threshold=0.35, store=store)
            admit, _s = gate.admit(1, {"smoke", "heat"})
            self.assertTrue(admit)
            store.put(1, b"fire drill")
            self.assertEqual(len(gate), 1)

            # The same event again is familiar while its trace lives.
            self.assertFalse(gate.admit(1, {"smoke", "heat"})[0])

            # Fill past capacity so the first trace is evicted.
            for key in (2, 3, 4):
                store.put(key, b"later")
            self.assertIsNone(store.tier(1))

            # Now it is novel again, and the dead key is gone.
            admit, salience = gate.admit(1, {"smoke", "heat"})
            self.assertTrue(admit)
            self.assertGreater(salience, 0.0)

    def test_without_a_store_the_gate_keeps_everything(self):
        gate = NoveltyGate(threshold=0.35)
        gate.admit(1, {"smoke", "heat"})
        self.assertFalse(gate.admit(1, {"smoke", "heat"})[0])
        self.assertEqual(len(gate), 1)


class NoveltyTest(unittest.TestCase):
    def setUp(self):
        self.gate = NoveltyGate(threshold=0.35)

    def test_first_event_always_novel(self):
        admit, salience = self.gate.admit(1, {"red", "loud"})
        self.assertTrue(admit)
        self.assertEqual(salience, 1.0)

    def test_repeat_is_not_encoded(self):
        self.gate.admit(1, {"red", "loud"})
        admit, _ = self.gate.admit(2, {"red", "loud"})
        self.assertFalse(admit)

    def test_distinct_event_is_encoded(self):
        self.gate.admit(1, {"red", "loud"})
        admit, salience = self.gate.admit(2, {"cold", "quiet"})
        self.assertTrue(admit)
        self.assertEqual(salience, 1.0)

    def test_one_shot_no_repetition_needed(self):
        self.gate.admit(1, {"a", "b", "c"})
        self.assertEqual(len(self.gate), 1)


class CaptureTest(unittest.TestCase):
    def test_weak_trace_rescued_by_nearby_salient_event(self):
        with Store() as store:
            store.put(1, "weak", salience=0.2)
            before = store.activation(1)
            store.put(2, "strong", salience=0.2)
            store.note_salient(2, 4.0)
            self.assertGreater(store.activation(1), before)
            self.assertEqual(store.stats()["captures"], 1)

    def test_expired_tag_is_not_captured(self):
        with Store() as store:
            store.put(1, "late", salience=0.2)
            for _ in range(store.config.capture_window + 1):
                store.tick()
            cold = store.activation(1)
            store.put(2, "strong", salience=0.2)
            store.note_salient(2, 4.0)
            self.assertEqual(store.activation(1), cold)


class ConsolidatorTest(unittest.TestCase):
    def setUp(self):
        self.store = Store(promote_after=3)
        self.addCleanup(self.store.close)
        self.consolidator = Consolidator(self.store)
        self.store.put(1, "correction", salience=0.2)

    def test_cost_falls_with_repetition(self):
        first = self.consolidator.cost(1)
        self.consolidator.applied(1)
        second = self.consolidator.cost(1)
        self.assertLess(second, first)

    def test_promotion_pins_and_reclassifies(self):
        make_recurring(self.store, 1)
        for _ in range(3):
            self.consolidator.applied(1)
        self.assertTrue(self.consolidator.is_habit(1))
        self.assertEqual(self.store.pathway(1), Pathway.PROCEDURAL)
        self.assertEqual(self.store.tier(1), Tier.ACTIVE)

    def test_habit_survives_disuse(self):
        make_recurring(self.store, 1)
        for _ in range(3):
            self.consolidator.applied(1)
        for _ in range(30):
            self.store.tick()
        self.assertEqual(self.store.tier(1), Tier.ACTIVE)

    def test_unknown_key_costs_full_deliberation(self):
        self.assertEqual(self.consolidator.cost(999), 1.0)


class CascadeTest(unittest.TestCase):
    def setUp(self):
        self.cascade = Cascade()

    def test_irreversible_action_needs_more_evidence(self):
        self.assertGreater(
            self.cascade.commit_threshold(0.1),
            self.cascade.commit_threshold(1.0),
        )

    def test_orienting_is_cheap_regardless_of_reversibility(self):
        self.assertEqual(
            self.cascade.stage(0.3, reversibility=0.01), Stage.ORIENT
        )

    def test_reversible_action_commits_on_strong_evidence(self):
        self.assertEqual(
            self.cascade.stage(0.9, reversibility=1.0, signals=2),
            Stage.COMMIT,
        )

    def test_same_evidence_will_not_commit_when_irreversible(self):
        self.assertEqual(
            self.cascade.stage(0.9, reversibility=0.05, signals=2),
            Stage.PREPARE,
        )

    def test_convergence_requirement_scales(self):
        self.assertGreater(
            self.cascade.required_signals(0.1),
            self.cascade.required_signals(1.0),
        )


class DispatcherTest(unittest.TestCase):
    def setUp(self):
        self.store = Store(promote_after=2)
        self.addCleanup(self.store.close)
        self.dispatcher = Dispatcher(self.store, Consolidator(self.store))

    def test_releaser_is_discrete(self):
        self.dispatcher.add_releaser("looming", 10, "flee")
        self.assertTrue(self.dispatcher.fires("looming"))
        self.assertFalse(self.dispatcher.fires("loomin"))

    def test_instinct_costs_nothing_and_never_misses(self):
        self.dispatcher.add_releaser("looming", 10, "flee")
        for _ in range(50):
            self.store.tick()
        outcome = self.dispatcher.act("looming")
        self.assertEqual(outcome.source, "instinct")
        self.assertEqual(outcome.cost, 0.0)
        self.assertEqual(self.store.tier(10), Tier.ACTIVE)

    def test_irreversible_releaser_will_not_commit_on_one_cue(self):
        self.dispatcher.add_releaser("prey", 11, "strike", reversibility=0.05)
        outcome = self.dispatcher.act("prey", signals=1)
        self.assertNotEqual(outcome.stage, Stage.COMMIT)

    def test_slow_pathway_revises_fast_action(self):
        self.dispatcher.add_releaser("stick", 12, "flee")
        outcome = self.dispatcher.act("stick", judge=lambda _d: (0.9, "stay"))
        self.assertEqual(outcome.source, "instinct")
        self.assertEqual(outcome.action, "flee")
        self.assertTrue(outcome.revised)
        self.assertEqual(outcome.revision, "stay")

    def test_declarative_route_when_no_releaser(self):
        self.store.put(20, "raw")
        outcome = self.dispatcher.act(20, judge=lambda _d: (0.9, "judged"))
        self.assertEqual(outcome.source, "declarative")
        self.assertEqual(outcome.action, "judged")

    def test_declarative_recall_rewrites_the_trace(self):
        self.store.put(21, "before")
        self.dispatcher.act(21, judge=lambda _d: (0.9, "after"))
        with self.store.lookup(21) as lease:
            self.assertEqual(lease.data, b"after")

    def test_habit_route_is_cheaper_than_deliberation(self):
        self.store.put(30, "shortcut")
        consolidator = Consolidator(self.store)
        make_recurring(self.store, 30)
        for _ in range(2):
            consolidator.applied(30)
        outcome = self.dispatcher.act(30)
        self.assertEqual(outcome.source, "procedural")
        self.assertLess(outcome.cost, 1.0)


class IntegrationTest(unittest.TestCase):
    def test_novel_event_encodes_spreads_and_consolidates(self):
        with Store(promote_after=2, active_k=4) as store:
            gate = NoveltyGate()
            graph = AssociativeGraph()
            consolidator = Consolidator(store)

            admit, salience = gate.admit(1, {"smoke", "heat"})
            self.assertTrue(admit)
            store.put(1, "fire drill", salience=salience)

            store.put(2, "exit route", salience=0.3)
            graph.link(1, 2, 0.9)

            warm = graph.prefetch_set([1])
            store.prefetch(warm)
            for key in warm:
                store.excite(key, graph.spread([1])[key])

            with store.lookup(2) as lease:
                self.assertEqual(lease.data, b"exit route")
            self.assertEqual(store.stats()["prefetch_hits"], 1)

            make_recurring(store, 2)
            for _ in range(2):
                consolidator.applied(2)
            self.assertTrue(consolidator.is_habit(2))

            for _ in range(20):
                store.tick()
            self.assertEqual(store.tier(2), Tier.ACTIVE)


class TruthValueSafeguardTest(unittest.TestCase):
    """Every one of these enums has a 0-valued member that would otherwise
    test False in the ordinary case, making `if result:` read backwards."""

    def test_written_is_truthy_despite_being_zero(self):
        self.assertEqual(int(Restabilize.WRITTEN), 0)
        self.assertTrue(Restabilize.WRITTEN)

    def test_gated_and_error_are_falsy(self):
        self.assertFalse(Restabilize.GATED)
        self.assertFalse(Restabilize.ERROR)

    def test_restabilize_result_reads_correctly_in_a_condition(self):
        with Store() as store:
            store.put(1, "old")
            store.retrieve(1).release()
            self.assertTrue(store.restabilize(1, "new"))
        with Store(gate_tonic=0.0) as store:
            store.put(1, "old")
            store.retrieve(1).release()
            self.assertFalse(store.restabilize(1, "new"))

    def test_tier_refuses_truth_testing(self):
        with self.assertRaises(TypeError):
            bool(Tier.ACTIVE)

    def test_pathway_refuses_truth_testing(self):
        with self.assertRaises(TypeError):
            bool(Pathway.DECLARATIVE)

    def test_absence_is_still_detectable(self):
        with Store() as store:
            self.assertIsNone(store.tier(404))
            self.assertIsNone(store.pathway(404))
            store.put(1, "x")
            self.assertIs(store.tier(1), Tier.EPISODIC)


class LeaseExpiryTest(unittest.TestCase):
    def test_leaked_lease_does_not_deadlock_encoding(self):
        with Store(capacity=1, lease_max_ticks=3) as store:
            store.put(1, "held")
            store.lookup(1)  # deliberately never released
            self.assertFalse(store.put(2, "new"))

            for _ in range(4):
                store.tick()

            self.assertEqual(store.stats()["leases_revoked"], 1)
            self.assertTrue(store.put(2, "new"))

    def test_lease_age_is_not_writable_from_above(self):
        # The reclaim signal must be owned by the autonomic process. excite()
        # is the pathway layer's only write into a trace, and it must not be
        # able to postpone expiry.
        with Store(capacity=1, lease_max_ticks=2) as store:
            store.put(1, "held")
            store.lookup(1)
            for _ in range(3):
                store.excite(1, 100.0)
                store.tick()
            self.assertEqual(store.stats()["leases_revoked"], 1)


class GateTest(unittest.TestCase):
    def test_low_tone_discards_the_update_but_keeps_the_trace(self):
        with Store(gate_tonic=0.0) as store:
            store.put(1, "old")
            store.retrieve(1).release()
            self.assertIs(store.restabilize(1, "new"), Restabilize.GATED)
            with store.lookup(1) as lease:
                self.assertEqual(lease.data, b"old")
            self.assertEqual(store.stats()["updates_gated"], 1)

    def test_salience_opens_the_gate(self):
        with Store(gate_tonic=0.0) as store:
            store.put(1, "old")
            store.note_salient(1, 2.0)
            self.assertGreaterEqual(store.gate(), 0.5)
            store.retrieve(1).release()
            self.assertIs(store.restabilize(1, "new"), Restabilize.WRITTEN)
            with store.lookup(1) as lease:
                self.assertEqual(lease.data, b"new")

    def test_phasic_tone_decays_back_to_tonic(self):
        with Store(gate_tonic=0.0) as store:
            store.put(1, "x")
            store.note_salient(1, 4.0)
            for _ in range(20):
                store.tick()
            self.assertLess(store.gate(), 0.5)


class SurpriseTest(unittest.TestCase):
    def setUp(self):
        self.store = Store(promote_after=3)
        self.addCleanup(self.store.close)
        self.consolidator = Consolidator(self.store)
        self.store.put(1, "correction", salience=0.2)

    def test_repetition_alone_does_not_promote(self):
        for _ in range(6):
            self.consolidator.applied(1, surprise=0.9)
        self.assertEqual(self.store.reps(1), 6)
        self.assertFalse(self.consolidator.is_habit(1))

    def test_unsurprising_repetition_promotes(self):
        make_recurring(self.store, 1)
        for _ in range(3):
            self.consolidator.applied(1, surprise=0.0)
        self.assertTrue(self.consolidator.is_habit(1))

    def test_surprise_resets_the_run(self):
        self.consolidator.applied(1, surprise=0.0)
        self.consolidator.applied(1, surprise=0.0)
        self.consolidator.applied(1, surprise=0.8)
        self.assertFalse(self.consolidator.is_habit(1))


class StochasticCompetitionTest(unittest.TestCase):
    def test_noise_makes_competition_a_draw(self):
        with Store() as store:
            act = [1.0, 1.05, 1.1, 1.15]
            outcomes = {
                tuple(store.kwta(act, 2, temp=0.6, seed=seed))
                for seed in range(1, 40)
            }
            self.assertGreater(len(outcomes), 1)

    def test_zero_temperature_is_deterministic(self):
        with Store() as store:
            act = [0.1, 0.9, 0.4, 0.8]
            self.assertEqual(store.kwta(act, 2, temp=0.0),
                             [False, True, False, True])

    def test_same_seed_reproduces_the_draw(self):
        with Store() as store:
            act = [0.2, 0.4, 0.6, 0.8, 1.0]
            self.assertEqual(store.kwta(act, 2, temp=0.7, seed=42),
                             store.kwta(act, 2, temp=0.7, seed=42))


class ConstitutionTest(unittest.TestCase):
    """The constitution supplies cost; it never picks an action."""

    def setUp(self):
        self.con = Constitution(
            costs={"note": 1.0, "publish": 0.3, "delete_backups": 0.05},
            forbidden=("erase_history",),
        )

    def test_declared_cost_is_returned_verbatim(self):
        self.assertEqual(self.con.reversibility("publish"), 0.3)

    def test_unclassified_act_is_expensive_not_free(self):
        """Silence is not permission: an unnamed act costs the most."""
        self.assertEqual(self.con.reversibility("something_new"), 0.05)
        self.assertLess(
            self.con.reversibility("something_new"),
            self.con.reversibility("note"),
        )

    def test_cost_raises_the_bar_the_cascade_demands(self):
        """The whole mechanism: values never choose, they make acts steep."""
        cascade = Cascade()
        cheap = self.con.reversibility("note")
        grave = self.con.reversibility("delete_backups")
        self.assertLess(
            cascade.commit_threshold(cheap),
            cascade.commit_threshold(grave),
        )
        self.assertLess(
            cascade.required_signals(cheap),
            cascade.required_signals(grave),
        )

    def test_evidence_commits_a_cheap_act_but_not_a_grave_one(self):
        cascade = Cascade()
        evidence, signals = 0.9, 4
        self.assertIs(
            cascade.stage(evidence, self.con.reversibility("note"), signals),
            Stage.COMMIT,
        )
        self.assertIsNot(
            cascade.stage(evidence,
                          self.con.reversibility("delete_backups"),
                          signals),
            Stage.COMMIT,
        )

    def test_forbidden_is_categorical_not_scaled(self):
        """No evidence clears it, unlike a cost which enough evidence does."""
        self.assertFalse(self.con.permits("erase_history"))
        self.assertTrue(self.con.permits("publish"))
        self.assertTrue(self.con.permits("never_mentioned"))

    def test_forbidden_act_may_not_also_carry_a_cost(self):
        """A price on a refusal invites the reading that evidence buys it."""
        with self.assertRaises(ValueError):
            Constitution(costs={"x": 0.5}, forbidden=("x",))

    def test_declared_costs_are_validated(self):
        with self.assertRaises(ValueError):
            Constitution(costs={"x": 1.5})

    def test_the_bound_layer_cannot_rewrite_what_binds_it(self):
        """No setter, and the stored mappings are read-only proxies."""
        with self.assertRaises(TypeError):
            self.con.costs["publish"] = 1.0
        with self.assertRaises(AttributeError):
            self.con.costs = {}
        self.assertNotIn("add", dir(self.con.forbidden))

    def test_construction_copies_so_later_edits_do_not_leak_in(self):
        source = {"note": 1.0}
        con = Constitution(costs=source)
        source["note"] = 0.0
        self.assertEqual(con.reversibility("note"), 1.0)


class SpreadFloorIsDerived(unittest.TestCase):
    """The cutoff was 0.01, a number with no relation to the graph it
    cut. It is now decay**depth -- the weakest contribution any
    legitimate path can deliver -- so it moves with the two parameters
    that define reach.
    """

    def chain(self, **kw):
        g = AssociativeGraph(**kw)
        g.link(1, 2, 0.9)
        g.link(2, 3, 0.9)
        g.link(3, 4, 0.9)
        return g

    def test_the_floor_is_the_weakest_legitimate_path(self):
        g = AssociativeGraph()
        self.assertAlmostEqual(g.floor, g.decay ** g.depth)

    def test_a_hop_weaker_than_any_full_path_is_not_followed(self):
        # 0.9-weighted hops at decay 0.5 reach 0.091 by the third hop,
        # under the derived floor of 0.125 and over the old 0.01.
        reached = self.chain()
        self.assertNotIn(4, reached.spread([1]),
                         "a contribution weaker than the weakest whole "
                         "path was still followed")

    def test_an_explicit_floor_still_wins(self):
        self.assertIn(4, self.chain(floor=0.01).spread([1]))

    def test_the_floor_follows_a_depth_override(self):
        # Reaching further makes the weakest legitimate path weaker, so
        # the cutoff has to move with it or it is just a constant again.
        g = self.chain()
        self.assertIn(4, g.spread([1], depth=6))


if __name__ == "__main__":
    unittest.main(verbosity=1)
