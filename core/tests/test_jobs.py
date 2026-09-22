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


if __name__ == "__main__":
    unittest.main()
