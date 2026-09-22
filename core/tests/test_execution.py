"""execution: order, placement, the device lock and resource lines."""

import os
import sys
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "py"))

from locus import execution  # noqa: E402


def square(x):
    return x * x


def slow(x):
    time.sleep(0.05)
    return x


class TestExecution(unittest.TestCase):

    def test_results_follow_the_specs(self):
        specs = [3, 1, 2, 5, 4]
        self.assertEqual(execution.map_jobs(square, specs, procs=4),
                         [9, 1, 4, 25, 16])

    def test_serial_and_parallel_agree(self):
        specs = list(range(9))
        self.assertEqual(execution.map_jobs(square, specs, procs=1),
                         execution.map_jobs(square, specs, procs=4))

    def test_empty_and_single(self):
        self.assertEqual(execution.map_jobs(square, []), [])
        self.assertEqual(execution.map_jobs(square, [6]), [36])

    def test_cores_never_exceeds_the_job_count(self):
        self.assertEqual(execution.cores(2, procs=16), 2)
        self.assertEqual(execution.cores(3, procs=2), 2)
        self.assertGreaterEqual(execution.cores(4), 1)

    def test_device_lock_is_one_object(self):
        self.assertIs(execution.device_lock(), execution.device_lock())

    def test_device_context_counts_its_time(self):
        before = getattr(execution.device, "seconds", 0.0)
        with execution.device():
            time.sleep(0.02)
        self.assertGreater(execution.device.seconds, before)

    def test_run_binary_raises_on_failure(self):
        with self.assertRaises(RuntimeError):
            execution.run_binary([sys.executable, "-c", "raise SystemExit(3)"])
        r = execution.run_binary([sys.executable, "-c", "print('ok')"])
        self.assertEqual(r.stdout.strip(), "ok")

    def test_resources_measures_and_formats(self):
        with execution.resources() as res:
            execution.map_jobs(slow, list(range(4)), procs=2)
            res.device += 1.5
        self.assertGreater(res.wall, 0.0)
        line = res.line(procs=2)
        self.assertIn("on 2 processes", line)
        self.assertIn("GPU device 1.5 s", line)


if __name__ == "__main__":
    unittest.main()
