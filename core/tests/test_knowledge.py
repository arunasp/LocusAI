"""Knowledge store: format, readout and learning while reading."""

import json
import math
import os
import sys
import tempfile
import unittest
from array import array

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "py"))

from locus.encode import NgramEncoder  # noqa: E402
from locus.knowledge import Knowledge, MAGIC  # noqa: E402

SAMPLE = [b"abcabcabd", b"abcd"]


def write_store(path, enc, stores, rate=0.1, lam=0.5, sleep=1e9,
                readout="plain"):
    """stores: list of (flags, {unit: {byte: weight}}, {unit: events}).
    `sleep` is the sidecar's sleep threshold; the default never fires,
    so tests of the tiers themselves see no autonomic sleep."""
    U = enc.n
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(array("i", [U, len(stores)]).tobytes())
        f.write(array("d", [rate]).tobytes())
        for flags, rows, events in stores:
            f.write(array("i", [flags]).tobytes())
            f.write(array("q", [events.get(u, 0) for u in range(U)])
                    .tobytes())
            rp, by, w = [0], bytearray(), array("d")
            for u in range(U):
                for b in sorted(rows.get(u, {})):
                    by.append(b)
                    w.append(rows[u][b])
                rp.append(len(w))
            f.write(array("q", rp).tobytes())
            f.write(bytes(by))
            f.write(w.tobytes())
    with open(path + ".json", "w") as fh:
        json.dump({"orders": list(enc.orders), "heads": enc.heads,
                   "tables": [list(t) for t in enc.tables], "lam": lam,
                   "learner": "test", "test_bpb": 0.0, "train_bytes": 0,
                   "sleep_threshold": sleep, "readout": readout},
                  fh)


class TestKnowledge(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "k")
        self.enc = NgramEncoder(SAMPLE)

    def test_encoder_from_tables_reproduces_units(self):
        e2 = NgramEncoder.from_tables(self.enc.orders, self.enc.heads,
                                      self.enc.tables)
        self.assertEqual(e2.n, self.enc.n)
        for data in SAMPLE:
            for pos in range(len(data)):
                self.assertEqual(e2.units_at(data, pos),
                                 self.enc.units_at(data, pos))

    def test_readout_sums_stores_and_clips(self):
        a = ord("a")
        write_store(self.path, self.enc, [
            (0, {a: {ord("b"): 0.75, ord("c"): -0.5}}, {a: 3}),
        ])
        k = Knowledge(self.path)
        d = k.drive(b"a", 0)
        self.assertEqual(d, {ord("b"): 0.75})
        self.assertAlmostEqual(
            k.bits(d, ord("b")),
            -math.log2((0.75 + 0.5 / 256) / (0.75 + 0.5)), places=12)

    def test_top_only_store_sees_top_order_units(self):
        k_units = self.enc.units_at(b"abcd", 3)
        top = [u for u, t in zip(k_units[1:], self.enc.tables)
               if t[0] == max(self.enc.orders)]
        write_store(self.path, self.enc, [
            (0, {}, {}),
            (3, {u: {ord("x"): 1.0} for u in top}, {}),
        ])
        k = Knowledge(self.path)
        self.assertEqual(k.drive(b"abcd", 3), {ord("x"): 1.0 * len(top)})
        self.assertEqual(k.drive(b"abcd", 1), {})

    def test_learning_continues_the_metaplastic_count(self):
        a, b = ord("a"), ord("b")
        write_store(self.path, self.enc, [(1, {a: {b: 0.5}}, {a: 1})])
        k = Knowledge(self.path)
        s = k.stores[0]
        s.learn(a, b, k.rate, 1)
        # second event: loc = 1 / (rate * 2), dw = rate * err * loc
        self.assertAlmostEqual(s.row(a)[b], 0.5 + 0.5 / 2, places=15)
        self.assertEqual(s.events[a], 2)

    def test_reading_without_learning_changes_nothing(self):
        write_store(self.path, self.enc, [
            (1, {ord("a"): {ord("b"): 0.5}}, {ord("a"): 1})])
        k = Knowledge(self.path)
        first = k.read(b"abab", learn=False)
        self.assertEqual(k.read(b"abab", learn=False), first)
        self.assertEqual(k.stores[0].changed, {})

    def test_learning_lowers_surprise_on_repeat(self):
        write_store(self.path, self.enc, [(1, {}, {})])
        k = Knowledge(self.path)
        first = sum(k.read(b"abcabc"))
        self.assertLess(sum(k.read(b"abcabc")), first)

    def test_rejects_a_mismatched_store(self):
        write_store(self.path, self.enc, [(0, {}, {})])
        with open(self.path, "r+b") as f:
            f.write(b"NOTLOCUS")
        with self.assertRaises(ValueError):
            Knowledge(self.path)


class TestConsolidation(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "k")
        self.enc = NgramEncoder(SAMPLE)
        write_store(self.path, self.enc, [
            (1, {ord("a"): {ord("b"): 0.5}}, {ord("a"): 1}), (3, {}, {})])

    def test_capture_persists_what_was_read(self):
        k = Knowledge(self.path)
        k.read(b"abcabd")
        rows = [dict(s.changed) for s in k.stores]
        after = k.read(b"abcabd", learn=False)
        self.assertGreater(k.consolidate(1.0, "test"), 0)
        k2 = Knowledge(self.path)
        self.assertEqual(k2.meta["generation"], 1)
        self.assertEqual(k2.read(b"abcabd", learn=False), after)
        for s, r in zip(k2.stores, rows):
            for u, row in r.items():
                self.assertEqual(s.row(u),
                                 {b: w for b, w in row.items() if w})
        self.assertTrue(os.path.exists(self.path + ".prev"))
        self.assertTrue(os.path.exists(self.path + ".prev.json"))

    def test_nonpositive_outcome_lapses_exactly(self):
        k = Knowledge(self.path)
        before = k.read(b"abcabd", learn=False)
        events = [list(s.events) for s in k.stores]
        with open(self.path, "rb") as fh:
            disk = fh.read()
        k.read(b"abcabd")
        self.assertEqual(k.consolidate(0.0), 0)
        self.assertEqual(k.tagged(), 0)
        self.assertEqual([list(s.events) for s in k.stores], events)
        self.assertEqual(k.read(b"abcabd", learn=False), before)
        with open(self.path, "rb") as fh:
            self.assertEqual(fh.read(), disk)
        self.assertFalse(os.path.exists(self.path + ".prev"))

    def test_prev_is_the_previous_generation(self):
        with open(self.path, "rb") as fh:
            gen0 = fh.read()
        k = Knowledge(self.path)
        k.read(b"abab")
        k.consolidate(1.0)
        with open(self.path + ".prev", "rb") as fh:
            self.assertEqual(fh.read(), gen0)


class TestGatedReadout(unittest.TestCase):
    """Each row scaled by its agreement with the sum of the others."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.enc = NgramEncoder(SAMPLE)

    def store(self, readout):
        path = os.path.join(self.dir, readout)
        a, b, c = ord("a"), ord("b"), ord("c")
        write_store(path, self.enc, [
            (0, {a: {b: 1.0, c: 0.5}}, {}),
            (3, {u: {b: 1.0} for u in self.top()}, {}),
        ], readout=readout)
        return Knowledge(path)

    def top(self):
        units = self.enc.units_at(b"abcd", 3)
        return [u for u, t in zip(units[1:], self.enc.tables)
                if t[0] == max(self.enc.orders)]

    def test_gate_scales_each_row_by_its_agreement(self):
        k = self.store("gated")
        rows = [r for s, units in k.active(b"abcd", 3)
                for u in units for r in [s.row(u)] if r]
        total = {}
        for r in rows:
            for b, w in r.items():
                total[b] = total.get(b, 0.0) + w
        want = {}
        for r in rows:
            rest = {b: total[b] - r.get(b, 0.0) for b in total}
            g = k.gate(r, rest)
            for b, w in r.items():
                want[b] = want.get(b, 0.0) + g * w
        got = k.drive(b"abcd", 3)
        self.assertEqual(sorted(got), sorted(b for b in want if want[b] > 0))
        for b in got:
            self.assertAlmostEqual(got[b], want[b], places=15)
        self.assertNotAlmostEqual(got[ord("b")], total[ord("b")], places=6)

    def test_plain_readout_sums_rows(self):
        k = self.store("plain")
        self.assertFalse(k.gated)
        got = k.drive(b"abcd", 3)
        # the byte unit at position 3 is "d", so only the top-order rows
        # of the second store fire, one unit each
        self.assertAlmostEqual(got[ord("b")], float(len(self.top())),
                               places=12)


class TestTiers(unittest.TestCase):
    """Transient tier, delayed outcome, offline sleep."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "k")
        self.enc = NgramEncoder(SAMPLE)
        write_store(self.path, self.enc, [(1, {}, {}), (3, {}, {})])
        with open(self.path, "rb") as fh:
            self.gen0 = fh.read()

    def disk(self):
        with open(self.path, "rb") as fh:
            return fh.read()

    def test_transient_tier_is_seen_before_sleep(self):
        k = Knowledge(self.path)
        k.read(b"abcabd")
        after = k.read(b"abcabd", learn=False)
        k.save_transient()
        self.assertEqual(self.disk(), self.gen0)
        k2 = Knowledge(self.path)
        self.assertEqual(k2.read(b"abcabd", learn=False), after)
        self.assertEqual(k2.clock, 5)

    def test_no_outcome_lapses_at_sleep(self):
        k = Knowledge(self.path)
        k.read(b"abcabd")
        k.save_transient()
        captured, lapsed = Knowledge(self.path).sleep()
        self.assertEqual(captured, 0)
        self.assertGreater(lapsed, 0)
        self.assertEqual(self.disk(), self.gen0)
        self.assertFalse(os.path.exists(self.path + ".tags"))
        self.assertEqual(Knowledge(self.path).clock, 5)

    def test_outcome_in_a_later_process_captures(self):
        k = Knowledge(self.path)
        k.read(b"abcabd")
        after = k.read(b"abcabd", learn=False)
        k.save_transient()
        k2 = Knowledge(self.path)
        k2.outcome(1.0)
        k2.save_transient()
        captured, lapsed = Knowledge(self.path).sleep()
        self.assertGreater(captured, 0)
        self.assertEqual(lapsed, 0)
        k3 = Knowledge(self.path)
        self.assertEqual(k3.meta["generation"], 1)
        self.assertEqual(k3.read(b"abcabd", learn=False), after)

    def test_outcome_credits_only_earlier_tags(self):
        k = Knowledge(self.path)
        k.read(b"ab")                       # tags at tick 1
        k.outcome(1.0)                      # tick 1
        k.read(b"cd")                       # tags at tick 2, after it
        a, c = ord("a"), ord("c")
        captured, lapsed = k.sleep()
        self.assertGreater(captured, 0)
        self.assertGreater(lapsed, 0)
        k2 = Knowledge(self.path)
        self.assertIn(ord("b"), k2.stores[0].row(a))
        self.assertEqual(k2.stores[0].row(c), {})

    def test_lifetime_bounds_credit(self):
        k = Knowledge(self.path)
        k.read(b"ab")                       # tick 1
        k.read(b"xxxxxxxxxx")               # ticks 2-10
        k.outcome(1.0)                      # tick 10
        k.sleep(lifetime=3)
        k2 = Knowledge(self.path)
        self.assertEqual(k2.stores[0].row(ord("a")), {})
        self.assertIn(ord("x"), k2.stores[0].row(ord("x")))

    def test_negative_sum_lapses(self):
        k = Knowledge(self.path)
        k.read(b"abab")
        k.outcome(1.0)
        k.outcome(-2.0)
        captured, lapsed = k.sleep()
        self.assertEqual(captured, 0)
        self.assertGreater(lapsed, 0)
        self.assertEqual(self.disk(), self.gen0)


class TestAutonomic(unittest.TestCase):
    """LocusAI sleeps by itself when learning has built enough pressure."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "k")
        self.enc = NgramEncoder(SAMPLE)
        a = ord("a")
        write_store(self.path, self.enc,
                    [(1, {a: {ord("b"): 0.5}}, {a: 100}), (3, {}, {})])

    def test_pressure_is_new_events_over_lifetime(self):
        k = Knowledge(self.path, sleep_threshold=1e9)
        self.assertEqual(k.lifetime, 100)
        self.assertEqual(k.pressure(), 0.0)
        k.read(b"abc")
        new = sum(s.events[u] - s.prior[u] for s in k.stores
                  for u in s.changed)
        self.assertGreater(new, 0)
        self.assertEqual(k.pressure(), new / 100)

    def test_sleeps_by_itself_before_learning_more(self):
        k = Knowledge(self.path, sleep_threshold=0.01)
        k.read(b"abcabd")
        k.outcome(1.0)
        self.assertIsNone(k.last_sleep)
        self.assertGreaterEqual(k.pressure(), 0.01)
        k.read(b"xy")                        # no sleep call anywhere
        captured, lapsed = k.last_sleep
        self.assertGreater(captured, 0)
        self.assertEqual(Knowledge(self.path).meta["generation"], 1)

    def test_no_sleep_below_threshold(self):
        k = Knowledge(self.path, sleep_threshold=1e9)
        k.read(b"abcabd")
        k.outcome(1.0)
        k.read(b"xy")
        self.assertIsNone(k.last_sleep)
        self.assertNotIn("generation", Knowledge(self.path).meta)

    def test_evaluation_never_triggers_sleep(self):
        k = Knowledge(self.path, sleep_threshold=0.01)
        k.read(b"abcabd")
        k.outcome(1.0)
        k.read(b"xy", learn=False)
        self.assertIsNone(k.last_sleep)
        self.assertGreater(k.tagged(), 0)

    def test_pressure_survives_between_processes(self):
        k = Knowledge(self.path, sleep_threshold=1e9)
        k.read(b"abcabd")
        p = k.pressure()
        k.save_transient()
        self.assertEqual(Knowledge(self.path).pressure(), p)

    def test_experience_lengthens_wake(self):
        """The same reading builds less pressure in a store with more
        lifetime experience."""
        young = Knowledge(self.path, sleep_threshold=1e9)
        young.read(b"abcabd")
        old_path = os.path.join(self.dir, "old")
        a = ord("a")
        write_store(old_path, self.enc,
                    [(1, {a: {ord("b"): 0.5}}, {a: 10000}), (3, {}, {})])
        old = Knowledge(old_path, sleep_threshold=1e9)
        old.read(b"abcabd")
        self.assertLess(old.pressure(), young.pressure())


if __name__ == "__main__":
    unittest.main()
