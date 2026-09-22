"""hf_data.py against a local mock of the Hub dataset API (no network)."""

import hashlib
import http.server
import json
import os
import sys
import tempfile
import threading
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

import hf_data  # noqa: E402

BIG = b"story. " * 1000
FILES = {
    "README.md": b"---\nlicense: cdla-sharing-1.0\n---\n# corpus\n",
    "small.txt": b"once upon a time\n",
    "big.txt": BIG,
}
LFS = {"big.txt"}


def entries():
    out = []
    for p, d in FILES.items():
        e = {"type": "file", "path": p, "size": len(d)}
        if p in LFS:
            e["lfs"] = {"oid": hashlib.sha256(d).hexdigest(),
                        "size": len(d)}
        out.append(e)
    return out


ROWS = [{"row": {"text": "line %d" % i}} for i in range(250)]
FAIL_ROWS = [0]          # how many /rows requests still fail with 503


class Hub(http.server.BaseHTTPRequestHandler):

    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.startswith("/api/datasets/o/r/tree/main"):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(entries()).encode())
            return
        if self.path.startswith("/splits?"):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({"splits": [
                {"dataset": "o/r", "config": "default",
                 "split": "train"}]}).encode())
            return
        if self.path.startswith("/rows?"):
            if FAIL_ROWS[0] > 0:
                FAIL_ROWS[0] -= 1
                self.send_response(503)
                self.end_headers()
                return
            q = dict(p.split("=", 1) for p in
                     self.path.split("?", 1)[1].split("&"))
            off, ln = int(q["offset"]), int(q["length"])
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({
                "features": [{"name": "text"}],
                "rows": ROWS[off:off + ln],
                "num_rows_total": len(ROWS)}).encode())
            return
        prefix = "/datasets/o/r/resolve/main/"
        if self.path.startswith(prefix):
            data = FILES[self.path[len(prefix):]]
            rng = self.headers.get("Range")
            if rng:
                end = int(rng.split("-")[1])
                data = data[:end + 1]
                self.send_response(206)
            else:
                self.send_response(200)
            self.end_headers()
            self.wfile.write(data)
            return
        self.send_response(404)
        self.end_headers()


class TestHfData(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.srv = http.server.HTTPServer(("127.0.0.1", 0), Hub)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        hf_data.HUB = "http://127.0.0.1:%d" % cls.srv.server_port
        hf_data.ROWS = hf_data.HUB

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def setUp(self):
        self.dest = tempfile.mkdtemp()

    def manifest(self):
        with open(os.path.join(self.dest, "o/r/manifest.json")) as fh:
            return json.load(fh)

    def test_whole_file_is_hash_verified_and_licence_read(self):
        self.assertEqual(hf_data.main(
            ["o/r", self.dest, "--file", "big.txt"]), 0)
        m = self.manifest()
        self.assertEqual(m["license"], "cdla-sharing-1.0")
        rec = [f for f in m["files"] if f["path"] == "big.txt"][0]
        self.assertTrue(rec["verified"])
        self.assertFalse(rec["truncated"])
        with open(os.path.join(self.dest, "o/r/big.txt"), "rb") as fh:
            self.assertEqual(fh.read(), BIG)

    def test_oversize_file_is_fetched_as_a_prefix(self):
        cap = 100
        self.assertEqual(hf_data.main(
            ["o/r", self.dest, "--file", "big.txt",
             "--max-mb", str(cap / 2**20)]), 0)
        rec = [f for f in self.manifest()["files"]
               if f["path"] == "big.txt"][0]
        self.assertTrue(rec["truncated"])
        self.assertEqual(rec["bytes"], cap)
        self.assertEqual(rec["published_size"], len(BIG))
        self.assertNotIn("verified", rec)
        with open(os.path.join(self.dest, "o/r/big.txt"), "rb") as fh:
            self.assertEqual(fh.read(), BIG[:cap])

    def test_unknown_file_is_refused(self):
        self.assertEqual(hf_data.main(
            ["o/r", self.dest, "--file", "nope.txt"]), 2)
        self.assertFalse(os.path.exists(os.path.join(self.dest, "o/r")))

    def test_rows_mode_streams_every_row(self):
        self.assertEqual(hf_data.main(["o/r", self.dest, "--rows"]), 0)
        rec = [f for f in self.manifest()["files"]
               if f["path"] == "rows.txt"][0]
        self.assertEqual(rec["rows"], len(ROWS))
        self.assertEqual((rec["config"], rec["split"], rec["column"]),
                         ("default", "train", "text"))
        with open(os.path.join(self.dest, "o/r/rows.txt")) as fh:
            lines = fh.read().splitlines()
        self.assertEqual(lines[0], "line 0")
        self.assertEqual(lines[-1], "line %d" % (len(ROWS) - 1))

    def test_rows_mode_stops_at_the_byte_budget(self):
        cap = 30                       # bytes, a few lines
        self.assertEqual(hf_data.main(
            ["o/r", self.dest, "--rows", "--max-mb",
             str(cap / 2**20)]), 0)
        rec = [f for f in self.manifest()["files"]
               if f["path"] == "rows.txt"][0]
        self.assertLessEqual(rec["bytes"], cap)
        self.assertGreater(rec["rows"], 0)
        self.assertLess(rec["rows"], len(ROWS))
        size = os.path.getsize(os.path.join(self.dest, "o/r/rows.txt"))
        self.assertEqual(size, rec["bytes"])

    def test_server_errors_are_retried(self):
        hf_data.time.sleep = lambda *_: None      # no real backoff here
        FAIL_ROWS[0] = 2
        try:
            self.assertEqual(hf_data.main(["o/r", self.dest, "--rows"]), 0)
            rec = [f for f in self.manifest()["files"]
                   if f["path"] == "rows.txt"][0]
            self.assertEqual(rec["rows"], len(ROWS))
            self.assertNotIn("partial", rec)
            self.assertEqual(FAIL_ROWS[0], 0)
        finally:
            FAIL_ROWS[0] = 0

    def test_rows_kept_when_retries_run_out(self):
        hf_data.time.sleep = lambda *_: None
        FAIL_ROWS[0] = 99                # never recovers
        try:
            self.assertEqual(hf_data.main(["o/r", self.dest, "--rows"]), 0)
            rec = [f for f in self.manifest()["files"]
                   if f["path"] == "rows.txt"][0]
            self.assertIn("partial", rec)
            self.assertEqual(rec["rows"], 0)
        finally:
            FAIL_ROWS[0] = 0

    def test_licence_parser(self):
        self.assertEqual(hf_data.licence("---\nlicense: mit\n---\nx"), "mit")
        self.assertIsNone(hf_data.licence("# no front matter"))


if __name__ == "__main__":
    unittest.main()
