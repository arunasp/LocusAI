"""run_jobs: order, serial fallback, and every spec computed once."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))

import parallel  # noqa: E402


def square(x):
    return x * x


class TestRunJobs(unittest.TestCase):

    def test_results_follow_the_order_of_specs(self):
        specs = [3, 1, 2, 5, 4]
        self.assertEqual(parallel.run_jobs(square, specs, procs=4),
                         [9, 1, 4, 25, 16])

    def test_serial_path_gives_the_same_answer(self):
        specs = list(range(7))
        self.assertEqual(parallel.run_jobs(square, specs, procs=1),
                         parallel.run_jobs(square, specs, procs=4))

    def test_empty_and_single(self):
        self.assertEqual(parallel.run_jobs(square, []), [])
        self.assertEqual(parallel.run_jobs(square, [6]), [36])

    def test_cores_never_exceeds_the_job_count(self):
        self.assertEqual(parallel.cores([1, 2], procs=16), 2)
        self.assertEqual(parallel.cores([1, 2, 3], procs=2), 2)


if __name__ == "__main__":
    unittest.main()
