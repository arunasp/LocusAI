"""jobs: process trees, log state, records and the clean pass."""

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

import jobs  # noqa: E402


def proc(pid, ppid):
    return {"pid": pid, "ppid": ppid, "state": "S", "cpu": 0.0,
            "name": "x", "cmd": ""}


class TestTree(unittest.TestCase):

    def test_tree_returns_pid_and_every_descendant(self):
        procs = [proc(1, 0), proc(2, 1), proc(3, 2), proc(4, 1),
                 proc(9, 8)]
        self.assertEqual(sorted(jobs.tree(1, procs)), [1, 2, 3, 4])
        self.assertEqual(jobs.tree(9, procs), [9])

    def test_tree_of_an_unknown_pid_is_just_that_pid(self):
        self.assertEqual(jobs.tree(77, [proc(1, 0)]), [77])


class TestLogState(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def write(self, text):
        p = os.path.join(self.dir, "j.log")
        with open(p, "w") as fh:
            fh.write(text)
        return p

    def test_finished_log_reports_its_exit(self):
        self.assertEqual(jobs.log_state(self.write("a\nexit=0\n")), "exit=0")
        self.assertEqual(jobs.log_state(self.write("a\nexit=2\n\n")),
                         "exit=2")

    def test_unfinished_log_is_running(self):
        self.assertEqual(jobs.log_state(self.write("working\n")), "running")

    def test_missing_log(self):
        self.assertEqual(jobs.log_state(os.path.join(self.dir, "no")),
                         "no log")


class TestRecordsAndClean(unittest.TestCase):

    def setUp(self):
        self.root = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.root, "jobs"))

    def record(self, name, pid, log_text):
        log = os.path.join(self.root, name + ".log")
        with open(log, "w") as fh:
            fh.write(log_text)
        with open(os.path.join(self.root, "jobs", name + ".json"),
                  "w") as fh:
            json.dump({"job": name, "pid": pid, "log": log,
                       "cmd": "x", "start": time.time()}, fh)

    def test_records_are_read_back(self):
        self.record("a", 1, "exit=0\n")
        self.record("b", 2, "running\n")
        got = sorted(r["job"] for r in jobs.records(self.root))
        self.assertEqual(got, ["a", "b"])

    def test_clean_drops_finished_records_only(self):
        self.record("done", 999999, "exit=0\n")     # dead pid, finished
        self.record("busy", os.getpid(), "running\n")
        self.assertEqual(jobs.cmd_clean(self.root, jobs.all_procs()), 0)
        left = [r["job"] for r in jobs.records(self.root)]
        self.assertEqual(left, ["busy"])

    def test_stop_reports_when_a_job_is_already_gone(self):
        self.record("gone", 999999, "exit=0\n")
        self.assertEqual(
            jobs.cmd_stop(self.root, jobs.all_procs(), "gone"), 0)

    def test_stop_kills_a_real_tree(self):
        p = subprocess.Popen([sys.executable, "-c",
                              "import time; time.sleep(60)"])
        self.record("sleeper", p.pid, "running\n")
        rc = jobs.cmd_stop(self.root, jobs.all_procs(), "sleeper")
        self.assertEqual(rc, 0)
        self.assertIsNone(p.poll() and None)      # reap
        p.wait(timeout=5)


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
