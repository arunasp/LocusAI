"""Byte-stream encoder: unit layout, derived table sizes, determinism."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "py"))

from locus.encode import (BYTE_UNITS, NgramEncoder, held_out,  # noqa: E402
                          is_prime, next_prime)

TEXT = [b"the cat sat on the mat. the cat ran.", b"def f(x):\n    return x\n"]


class Primes(unittest.TestCase):

    def test_next_prime_is_smallest_prime_at_or_above(self):
        for x, want in ((0, 2), (2, 2), (4, 5), (14, 17), (97, 97)):
            self.assertEqual(next_prime(x), want)
        self.assertFalse(is_prime(91))


class Layout(unittest.TestCase):

    def setUp(self):
        self.enc = NgramEncoder(TEXT, orders=(2, 3), heads=2)

    def test_first_unit_is_the_byte_itself(self):
        data = TEXT[0]
        for pos in range(len(data)):
            self.assertEqual(self.enc.units_at(data, pos)[0], data[pos])

    def test_tables_are_prime_distinct_and_cover_distinct_ngrams(self):
        sizes = {}
        for k, _h, _seed, _off, size in self.enc.tables:
            self.assertTrue(is_prime(size))
            self.assertGreaterEqual(size, self.enc.distinct[k])
            sizes.setdefault(k, []).append(size)
        for k, s in sizes.items():
            self.assertEqual(len(set(s)), len(s), "heads share a size")

    def test_tables_are_disjoint_and_n_counts_every_unit(self):
        end = BYTE_UNITS
        for _k, _h, _seed, off, size in self.enc.tables:
            self.assertEqual(off, end)
            end = off + size
        self.assertEqual(self.enc.n, end)

    def test_units_fall_in_their_tables(self):
        data = TEXT[1]
        for pos in range(len(data)):
            units = self.enc.units_at(data, pos)
            ready = [t for t in self.enc.tables if pos >= t[0] - 1]
            self.assertEqual(len(units), 1 + len(ready))
            for u, (_k, _h, _s, off, size) in zip(units[1:], ready):
                self.assertTrue(off <= u < off + size)

    def test_same_ngram_gives_same_units_anywhere(self):
        a = b"xxthe"
        b = b"the"
        self.assertEqual(self.enc.units_at(a, 4)[1:],
                         self.enc.units_at(b, 2)[1:])

    def test_joint_collisions_never_exceed_any_single_head(self):
        per = {}
        for k, _h, _size, _d, shared in self.enc.collisions(TEXT):
            per[k] = min(per.get(k, shared), shared)
        for k, _d, joint in self.enc.joint_collisions(TEXT):
            self.assertLessEqual(joint, per[k])


class Split(unittest.TestCase):

    def test_held_out_is_stable_and_partial(self):
        paths = ["f%d.py" % i for i in range(200)]
        picked = [p for p in paths if held_out(p, 5)]
        self.assertEqual(picked, [p for p in paths if held_out(p, 5)])
        self.assertTrue(0 < len(picked) < len(paths))


if __name__ == "__main__":
    unittest.main()
