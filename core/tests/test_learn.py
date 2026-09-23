"""Stream learners and their readout (locus.learn)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "py"))

from locus.encode import BYTE_UNITS, NgramEncoder     # noqa: E402
from locus.learn import (BranchLearner, CLSLearner,        # noqa: E402
                         FaithfulLearner, TagLearner, drive_of, prob,
                         score, score_with, unit_order)
from locus.plasticity import Plasticity               # noqa: E402

STREAM = [b"abcabcabcabcabcabcabcabcabcabc"]


def encoder():
    return NgramEncoder(STREAM, orders=(2,), heads=1)


class Readout(unittest.TestCase):

    def test_probabilities_sum_to_one(self):
        drive = {97: 2.0, 98: 0.5, 99: -1.0}
        total = sum(prob(drive, b, 0.3) for b in range(BYTE_UNITS))
        self.assertAlmostEqual(total, 1.0)

    def test_empty_drive_is_uniform(self):
        self.assertAlmostEqual(prob({}, 5, 1.0), 1.0 / BYTE_UNITS)

    def test_score_counts_every_transition(self):
        enc = encoder()
        _bpb, n = score(Plasticity(), enc, STREAM, 1.0)
        self.assertEqual(n, len(STREAM[0]) - 1)


class Learners(unittest.TestCase):

    def check_learns(self, learner, p, enc):
        self.assertEqual(p.live_weights(), 0, "a weight was installed")
        learner.learn(STREAM)
        drive = drive_of(p, enc.units_at(b"xab", 2))
        self.assertGreater(prob(drive, ord("c"), 0.01), 0.5)
        before, _ = score(Plasticity(), enc, STREAM, 1.0)
        after, _ = score(p, enc, STREAM, 0.01)
        self.assertLess(after, before)

    def test_tag_learner_learns_the_sequence(self):
        enc, p = encoder(), Plasticity()
        self.check_learns(TagLearner(p, enc), p, enc)

    def test_faithful_learner_learns_the_sequence(self):
        enc, p = encoder(), Plasticity()
        self.check_learns(FaithfulLearner(p, enc), p, enc)

    def test_faithful_weights_stay_in_bounds(self):
        enc, p = encoder(), Plasticity()
        FaithfulLearner(p, enc, homeostasis=False).learn(STREAM)
        for w in p.weights.values():
            self.assertTrue(0.0 <= w <= 1.0)

    def test_branch_learner_learns_the_sequence(self):
        enc, p = encoder(), Plasticity()
        self.check_learns(BranchLearner(p, enc), p, enc)


class Metaplastic(unittest.TestCase):

    def test_rows_equal_running_frequencies(self):
        data = b"abacabadabacab"
        enc = NgramEncoder([data], orders=(2,), heads=1)
        p = Plasticity()
        BranchLearner(p, enc, metaplastic=True).learn([data])
        # after 'a' the stream shows b, c, b, d, b, c, b: b 4/7, c 2/7, d 1/7
        a = ord("a")
        self.assertAlmostEqual(p.weight(a, ord("b")), 4 / 7)
        self.assertAlmostEqual(p.weight(a, ord("c")), 2 / 7)
        self.assertAlmostEqual(p.weight(a, ord("d")), 1 / 7)

    def test_first_observation_is_learned_in_one_step(self):
        data = b"xy"
        enc = NgramEncoder([data], orders=(2,), heads=1)
        p = Plasticity()
        BranchLearner(p, enc, metaplastic=True).learn([data])
        self.assertAlmostEqual(p.weight(ord("x"), ord("y")), 1.0)


class ComplementarySystems(unittest.TestCase):

    def setUp(self):
        self.data = [b"the cat sat. the cat ran. the dog sat.",
                     b"a cat sat on the mat."]
        self.enc = NgramEncoder(self.data, orders=(2, 3), heads=1)

    def test_unit_order(self):
        u = self.enc.units_at(b"abc", 2)
        self.assertEqual([unit_order(self.enc, x) for x in u], [1, 2, 3])

    def test_hippocampus_holds_only_top_order_units(self):
        h, c = Plasticity(), Plasticity()
        CLSLearner(h, c, self.enc).learn(self.data)
        self.assertTrue(h.weights)
        for (u, _b) in h.weights:
            self.assertEqual(unit_order(self.enc, u), 3)

    def test_cortex_alone_equals_constant_rate_branch(self):
        c, b = Plasticity(), Plasticity()
        CLSLearner(Plasticity(), c, self.enc, replay=False,
                   hippocampus=False).learn(self.data)
        BranchLearner(b, self.enc).learn(self.data)
        self.assertEqual(c.weights, b.weights)

    def test_replay_changes_the_cortex(self):
        a, b = Plasticity(), Plasticity()
        CLSLearner(Plasticity(), a, self.enc, replay=True).learn(self.data)
        CLSLearner(Plasticity(), b, self.enc, replay=False).learn(self.data)
        self.assertNotEqual(a.weights, b.weights)

    def test_readout_sums_both_systems(self):
        h, c = Plasticity(), Plasticity()
        cls = CLSLearner(h, c, self.enc)
        cls.learn(self.data)
        units = self.enc.units_at(b"the cat", 6)
        hd = drive_of(h, cls.hippo_units(units))
        cd = drive_of(c, units)
        for byte, v in cls.drive(units).items():
            self.assertAlmostEqual(v, hd.get(byte, 0.0) + cd.get(byte, 0.0))
        bpb, n = score_with(cls.drive, self.enc, self.data, 0.1)
        self.assertEqual(n, sum(len(d) - 1 for d in self.data))


class LamIsAFloorNotAConstant(unittest.TestCase):
    """`p(b) = (drive[b] + lam / BYTE_UNITS) / (pos + lam)`.

    The 256 is load-bearing: summing the numerator over exactly
    BYTE_UNITS candidates gives pos + lam, which is the denominator, so
    this normalises only while the support is that size and the drive is
    filtered to it. lam is read from a store's JSON sidecar, so it is an
    INPUT rather than a constant.
    """

    drive = {65: 2.0, 66: 1.0}

    def test_no_byte_is_impossible(self):
        # What lam BUYS: an unseen byte still has probability, so -log2
        # of it is a number of bits rather than an error.
        for lam in (3.05176e-05, 1.0):
            for b in (0, 200, BYTE_UNITS - 1):
                self.assertGreater(prob(self.drive, b, lam), 0.0)

    def test_summing_to_one_does_not_prove_the_readout_is_sane(self):
        # WHY lam IS CHECKED RATHER THAN TRUSTED. At lam < 0 the mass
        # still sums to 1, so the obvious sanity check passes, while
        # single probabilities are NEGATIVE -- and the bits figure then
        # either raises or comes out better than physically possible.
        total = sum(prob(self.drive, b, -0.5)
                    for b in range(BYTE_UNITS))
        self.assertAlmostEqual(total, 1.0, places=12)
        self.assertLess(prob(self.drive, 200, -0.5), 0.0)

    def test_a_non_positive_or_non_finite_lam_is_refused(self):
        enc = encoder()
        for bad in (0.0, -0.5, float("nan"), float("inf")):
            with self.assertRaises(ValueError, msg="lam=%r" % (bad,)):
                score_with(lambda units: {}, enc, STREAM, bad)


if __name__ == "__main__":
    unittest.main()
