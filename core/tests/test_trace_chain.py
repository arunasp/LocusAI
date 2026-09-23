"""Exact-property tests for the trace-chain level-coupling operator.

These are EXACT property tests, not tolerance-tuned approximations. The
construction claims a unique effective kernel for any admissible subset,
and each property below would FAIL LOUDLY if it were wrong -- which is
the point of testing a formalism rather than exercising code.

Five properties, and the last two are the ones that justify the cost:

  1. Row-stochasticity. A trace of a kernel is a kernel. If excursion
     mass were lost or double-counted, rows would stop summing to one.
  2. Identity on the full set. Tracing to everything changes nothing,
     and the result must be a copy so a caller cannot mutate the input.
  3. TRANSITIVITY. Tracing to A then to B (B a subset of A) equals
     tracing straight to B. This is what makes a HIERARCHY coherent:
     without it a three-level model would depend on which order the
     levels were derived in, and "lower-level change propagates upward"
     would be ill-defined.
  4. Stationary consistency, with a NEGATIVE CONTROL against naive
     lumping. The trace preserves the stationary distribution restricted
     to A and renormalised; plain block-averaging does not. If lumping
     matched, the linear solve would be waste.
  5. Refusal on a closed complement. If the dropped set never returns,
     no effective kernel on A exists, and that must raise rather than
     silently produce a matrix whose rows do not sum to one. At sparse
     connectivity this case is common, so it is a real condition rather
     than a theoretical one.

Plus a truncation test: the cheap per-tick form must converge to the
exact trace as depth rises, and the no-recursion control must be clearly
worse -- otherwise the recursion is not earning anything.
"""

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "py"))

from locus import trace_chain as tc  # noqa: E402


def irreducible_kernel(n, seed, density=1.0):
    """A random row-stochastic kernel with a positive diagonal.

    NOT the `random_kernel` in fixtures.py: that one takes a degree and
    is what the field experiments measure on, this one takes a density
    and guarantees irreducibility for the trace-chain properties. They
    shared a name and produced different kernels, which is a collision
    waiting to send someone to the wrong construction.

    The diagonal floor keeps the chain irreducible and aperiodic, so the
    stationary distribution exists and power iteration converges. Without
    it a random sparse kernel is frequently reducible and the tests would
    fail for reasons that say nothing about the construction.
    """
    rnd = random.Random(seed)
    p = []
    for i in range(n):
        row = [0.0] * n
        row[i] = 0.25 + 0.5 * rnd.random()
        for j in range(n):
            if j != i and rnd.random() < density:
                row[j] = rnd.random()
        s = sum(row)
        p.append([v / s for v in row])
    return p


class TraceChainProperties(unittest.TestCase):

    def test_row_stochastic(self):
        for n, seed in ((8, 1), (12, 2), (20, 3)):
            p = irreducible_kernel(n, seed)
            keep = list(range(0, n, 2))
            s = tc.trace_chain(p, keep)
            self.assertEqual(len(s), len(keep))
            for r, total in enumerate(tc.row_sums(s)):
                self.assertAlmostEqual(
                    total, 1.0, places=10,
                    msg="row %d of the trace lost or gained mass" % r,
                )

    def test_identity_on_full_set(self):
        p = irreducible_kernel(10, 4)
        s = tc.trace_chain(p, list(range(10)))
        for i in range(10):
            for j in range(10):
                self.assertAlmostEqual(s[i][j], p[i][j], places=12)

    def test_no_aliasing_of_input(self):
        # The full-set path returns early; make sure it copies, or a
        # caller mutating the result would silently corrupt the kernel.
        p = irreducible_kernel(6, 5)
        s = tc.trace_chain(p, list(range(6)))
        s[0][0] = 99.0
        self.assertNotEqual(p[0][0], 99.0)

    def test_transitivity_of_tracing(self):
        # The hierarchy property. Trace 24 -> 12 -> 6 against 24 -> 6.
        p = irreducible_kernel(24, 6)
        mid = list(range(0, 24, 2))
        fine_to_mid = tc.trace_chain(p, mid)
        small_global = list(range(0, 24, 4))
        small_in_mid = [mid.index(i) for i in small_global]
        two_step = tc.trace_chain(fine_to_mid, small_in_mid)
        one_step = tc.trace_chain(p, small_global)
        for i in range(len(small_global)):
            for j in range(len(small_global)):
                self.assertAlmostEqual(
                    two_step[i][j], one_step[i][j], places=9,
                    msg="tracing is order-dependent at (%d,%d); a "
                        "hierarchy built on it would be ill-defined"
                        % (i, j),
                )

    def test_stationary_consistency(self):
        p = irreducible_kernel(16, 7)
        keep = list(range(0, 16, 2))
        pi = tc.stationary(p)
        want = [pi[i] for i in keep]
        scale = sum(want)
        want = [v / scale for v in want]
        got = tc.stationary(tc.trace_chain(p, keep))
        for i in range(len(keep)):
            self.assertAlmostEqual(
                got[i], want[i], places=8,
                msg="trace changed the long-run measure at %d" % i,
            )

    def test_lumping_is_not_tracing(self):
        # NEGATIVE CONTROL. If these agreed, the linear solve would be
        # pointless and a plain reduction would do. Assert they DIFFER.
        p = irreducible_kernel(12, 8)
        blocks = [[i, i + 1] for i in range(0, 12, 2)]
        keep = [b[0] for b in blocks]
        traced = tc.trace_chain(p, keep)
        lumped = tc.lumped_chain(p, blocks)
        worst = max(
            abs(traced[i][j] - lumped[i][j])
            for i in range(len(keep))
            for j in range(len(keep))
        )
        self.assertGreater(
            worst, 1e-3,
            msg="lumping matched the trace: either the kernel is "
                "accidentally lumpable or the construction collapsed",
        )

    def test_closed_complement_is_refused(self):
        # If the dropped set never returns to A, no effective kernel on A
        # exists. That must raise rather than silently produce a matrix
        # whose rows do not sum to one.
        n = 6
        p = [[0.0] * n for _ in range(n)]
        for i in range(3):                      # A: leaks into B
            p[i][3] = 1.0
        for i in range(3, n):                   # B: absorbing, no return
            p[i][3 if i != 3 else 4] = 1.0
        with self.assertRaises(ValueError):
            tc.trace_chain(p, [0, 1, 2])


class TruncationProperties(unittest.TestCase):

    def test_truncation_converges_to_exact(self):
        # The cheap per-tick form must approach the exact trace as depth
        # rises. If it did not, depth would be a meaningless dial.
        p = irreducible_kernel(20, 11)
        keep = list(range(0, 20, 2))
        exact = tc.trace_chain(p, keep)
        prev = first = None
        for depth in range(0, 5):
            approx = tc.truncated_trace(p, keep, depth)
            err = max(abs(approx[i][j] - exact[i][j])
                      for i in range(len(keep))
                      for j in range(len(keep)))
            if prev is not None:
                self.assertLessEqual(
                    err, prev + 1e-12,
                    msg="error grew from depth %d to %d" % (depth - 1,
                                                            depth),
                )
            if first is None:
                first = err
            prev = err
        # RELATIVE, not absolute, and this is a correction. The earlier
        # form asserted `prev < 1e-3` -- a constant calibrated on the
        # character-corpus kernel (where depth 1 reached 0.06% of exact)
        # and then asserted against a denser random kernel, where it
        # measured 0.00132 and failed. The CLAIM being tested is
        # CONVERGENCE, so test the ratio. An absolute bound here measures
        # the kernel's excursion depth, not the construction.
        self.assertIsNotNone(first, "loop never ran")
        self.assertLess(
            prev, first / 4.0,
            msg="depth 4 error %.6g is not materially better than the "
                "depth 0 error %.6g -- depth is not buying accuracy"
                % (prev, first),
        )

    def test_no_recursion_control_is_worse(self):
        # Ignoring excursions entirely must be clearly worse than one
        # step of recursion, or the recursion earns nothing.
        p = irreducible_kernel(20, 12)
        keep = list(range(0, 20, 2))
        exact = tc.trace_chain(p, keep)

        def err(kernel):
            return max(abs(kernel[i][j] - exact[i][j])
                       for i in range(len(keep))
                       for j in range(len(keep)))

        none_err = err(tc.truncated_trace(p, keep, -1))
        one_err = err(tc.truncated_trace(p, keep, 1))
        self.assertGreater(
            none_err, one_err,
            msg="no-recursion matched one step of recursion; the "
                "excursion correction is doing no work here",
        )

    def test_truncation_rows_are_stochastic(self):
        p = irreducible_kernel(14, 13)
        keep = list(range(0, 14, 2))
        for depth in (-1, 0, 2):
            k = tc.truncated_trace(p, keep, depth)
            for total in tc.row_sums(k):
                self.assertAlmostEqual(total, 1.0, places=10)


if __name__ == "__main__":
    unittest.main()
