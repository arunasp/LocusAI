"""fit_lam extends its grid until the best lam is interior."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))

import exp_learn_stream as X  # noqa: E402


class TestFitLam(unittest.TestCase):

    def test_interior_minimum_is_found_without_extending(self):
        calls = []

        def scorer(x):
            calls.append(x)
            return abs(x - 4.0)

        best, lam, edge = X.fit_lam(scorer)
        self.assertEqual(lam, 4.0)
        self.assertFalse(edge)
        self.assertEqual(len(calls), 17)     # the initial grid only

    def test_grid_extends_below_its_floor(self):
        best, lam, edge = X.fit_lam(lambda x: abs(x - 2.0 ** -20))
        self.assertEqual(lam, 2.0 ** -20)
        self.assertFalse(edge)

    def test_grid_extends_above_its_ceiling(self):
        best, lam, edge = X.fit_lam(lambda x: abs(x - 2.0 ** 12))
        self.assertEqual(lam, 2.0 ** 12)
        self.assertFalse(edge)

    def test_unbounded_minimum_reports_the_edge(self):
        best, lam, edge = X.fit_lam(lambda x: -x, limit=4)
        self.assertTrue(edge)

    def test_ties_take_the_smaller_exponent(self):
        best, lam, edge = X.fit_lam(lambda x: 1.0)
        self.assertEqual(lam, 2.0 ** -12)
        self.assertFalse(edge)


if __name__ == "__main__":
    unittest.main()
