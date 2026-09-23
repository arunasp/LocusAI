"""The job monitor must match processes, not filenames.

`jobs.py clean` reports processes that look like our workloads but
belong to no recorded job. It tested the whole command line for
substrings, so `make commit-verified FILES=core/tools/perfmon.py`
matched -- the pipeline that INVOKES the monitor reported itself, and a
real stray process would have sat two lines away from being missed.

Every case below is a command that actually occurred in this repo.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

import jobs  # noqa: E402


class NameInTheProgramPosition(unittest.TestCase):
    """A name in an argument is a file; a name in the program position
    is a process."""

    def test_our_own_tooling_is_not_a_workload(self):
        # Both of these were reported as unmanaged workloads, by the
        # very command that ran the monitor.
        self.assertFalse(jobs.is_workload(
            "make commit-verified FILES=core/tools/perfmon.py"))
        self.assertFalse(jobs.is_workload(
            "python3 cicd/gitops.py commit-verified --msg x -- "
            "core/tools/perfmon.py"))

    def test_naming_a_file_is_not_running_it(self):
        self.assertFalse(jobs.is_workload("grep -r know.py ."))
        self.assertFalse(jobs.is_workload("vim tools/know.py"))
        self.assertFalse(jobs.is_workload(
            "python3 -m pycodestyle tools/know.py"))

    def test_a_real_workload_is_still_matched(self):
        self.assertTrue(jobs.is_workload(
            "build/learn_device /tmp/a.in /tmp/a.out build/know/k"))
        self.assertTrue(jobs.is_workload(
            "/workspace/core/build/babble_device in out"))
        self.assertTrue(jobs.is_workload(
            "python3 tools/perfmon.py record build/x.perf.csv"))
        self.assertTrue(jobs.is_workload(
            ".venv/bin/python3 -u tools/know.py build/corpus "
            "build/learn_device build/know/k"))

    def test_make_bg_is_found_after_its_options(self):
        # The target sits after -C, so checking one position misses it.
        self.assertTrue(jobs.is_workload(
            "make -C /workspace/core bg JOB=x CMD=y"))
        self.assertFalse(jobs.is_workload("make -C /workspace/core all"))


class TheMonitorDoesNotReportItself(unittest.TestCase):

    def test_ancestry_walks_to_the_root(self):
        procs = [{"pid": 4, "ppid": 3, "cmd": "c"},
                 {"pid": 3, "ppid": 2, "cmd": "b"},
                 {"pid": 2, "ppid": 0, "cmd": "a"}]
        self.assertEqual(jobs.ancestry(4, procs), [4, 3, 2])

    def test_a_cycle_terminates(self):
        # /proc is read without locking, so a pid can be reused between
        # reads and the parent chain can appear to loop.
        procs = [{"pid": 1, "ppid": 2, "cmd": "a"},
                 {"pid": 2, "ppid": 1, "cmd": "b"}]
        self.assertEqual(sorted(jobs.ancestry(1, procs)), [1, 2])


if __name__ == "__main__":
    unittest.main()
