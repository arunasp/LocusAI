"""perfmon summary arithmetic on a synthetic sample file."""

import contextlib
import io
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

import perfmon  # noqa: E402


class Summary(unittest.TestCase):

    def run_summary(self, rows, device=None):
        fd, path = tempfile.mkstemp(suffix=".csv")
        with os.fdopen(fd, "w") as fh:
            fh.write("t,busy_cores,ncpu,mem_mib\n")
            for i, (b, m) in enumerate(rows, 1):
                fh.write("%d,%s,4,%d\n" % (i, b, m))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = perfmon.summary(path, device)
        os.unlink(path)
        return rc, out.getvalue()

    def test_mean_idle_share_and_core_seconds(self):
        # 4 CPUs, 4 one-second samples: 4, 4, 0.5, 0.5 busy cores.
        rc, text = self.run_summary(
            [("4", 10), ("4", 30), ("0.5", 20), ("0.5", 20)], device=1.0)
        self.assertEqual(rc, 0)
        self.assertIn("busy cores mean 2.2 (56%)", text)
        self.assertIn("below 25% of CPUs for 50% of the time", text)
        self.assertIn("CPU core-seconds 9 of 16 available", text)
        self.assertIn("peak memory 30 MiB", text)
        self.assertIn("GPU device 1.0 s = 25.0% of wall", text)

    def test_empty_file_fails(self):
        rc, text = self.run_summary([])
        self.assertEqual(rc, 1)
        self.assertIn("no samples", text)


if __name__ == "__main__":
    unittest.main()
