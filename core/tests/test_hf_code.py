"""hf_code.py against a local mock of the Hub API (no network)."""

import hashlib
import http.server
import json
import os
import sys
import tempfile
import threading
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

import hf_code  # noqa: E402

FILES = {
    "config.json": b'{"n_layers": 2}',
    "inference/model.py": b"def forward(x):\n    return x\n",
    "model-00001.safetensors": b"\0" * 64,
    "big.txt": b"x" * 3000,
}
LFS = {"model-00001.safetensors"}


def blob_sha1(data):
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def entries():
    out = []
    for p, d in FILES.items():
        e = {"type": "file", "path": p, "size": len(d), "oid": blob_sha1(d)}
        if p in LFS:
            e["lfs"] = {"oid": hashlib.sha256(d).hexdigest(), "size": len(d)}
        out.append(e)
    return out


class Hub(http.server.BaseHTTPRequestHandler):
    tamper = False

    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.startswith("/api/models/o/r/tree/main"):
            es = entries()
            page2 = "cursor=2" in self.path
            body = json.dumps(es[2:] if page2 else
                              es[:2] + [{"type": "directory",
                                         "path": "inference"}]).encode()
            self.send_response(200)
            if not page2:
                self.send_header("Link", '<http://127.0.0.1:%d/api/models/o/'
                                 'r/tree/main?recursive=true&cursor=2>; '
                                 'rel="next"' % self.server.server_port)
            self.end_headers()
            self.wfile.write(body)
            return
        prefix = "/o/r/resolve/main/"
        if self.path.startswith(prefix):
            data = FILES[self.path[len(prefix):]]
            if Hub.tamper:
                data = data + b"!"
            self.send_response(200)
            self.end_headers()
            self.wfile.write(data)
            return
        self.send_response(404)
        self.end_headers()


class TestHfCode(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.srv = http.server.HTTPServer(("127.0.0.1", 0), Hub)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        hf_code.HUB = "http://127.0.0.1:%d" % cls.srv.server_port

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def setUp(self):
        Hub.tamper = False
        self.dest = tempfile.mkdtemp()

    def test_keeps_code_skips_tensors_and_oversize(self):
        m = hf_code.fetch("o/r", self.dest, max_mb=2000 / 2**20)
        self.assertEqual(sorted(f["path"] for f in m["kept"]),
                         ["config.json", "inference/model.py"])
        why = {f["path"]: f["reason"] for f in m["skipped"]}
        self.assertEqual(why, {"model-00001.safetensors": "tensor file",
                               "big.txt": "over size cap"})
        root = os.path.join(self.dest, "o/r")
        for p in ("config.json", "inference/model.py"):
            with open(os.path.join(root, p), "rb") as fh:
                self.assertEqual(fh.read(), FILES[p])
        self.assertFalse(os.path.exists(
            os.path.join(root, "model-00001.safetensors")))
        with open(os.path.join(root, "manifest.json")) as fh:
            self.assertEqual(len(json.load(fh)["skipped"]), 2)

    def test_dry_run_downloads_nothing(self):
        m = hf_code.fetch("o/r", self.dest, dry_run=True)
        self.assertEqual(len(m["kept"]), 3)
        self.assertEqual(os.listdir(self.dest), [])

    def test_hash_mismatch_is_refused(self):
        Hub.tamper = True
        with self.assertRaises(ValueError):
            hf_code.fetch("o/r", self.dest)

    def test_lfs_files_verify_by_sha256(self):
        e = [x for x in entries() if x["path"] in LFS][0]
        self.assertTrue(hf_code.verify(e, FILES[e["path"]]))
        self.assertFalse(hf_code.verify(e, FILES[e["path"]] + b"x"))


if __name__ == "__main__":
    unittest.main()
