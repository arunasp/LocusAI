"""chunk_text: record boundaries, packing and the round trip."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

import chunk_text  # noqa: E402

SEP = b"<|endoftext|>"


class TestChunkText(unittest.TestCase):

    def test_records_drop_empty_and_keep_content(self):
        data = SEP.join([b"one", b"", b"  ", b"two"])
        self.assertEqual(chunk_text.records(data, SEP), [b"one", b"two"])

    def test_pack_keeps_records_whole_and_loses_none(self):
        recs = [b"x" * 30 for _ in range(10)]
        chunks = chunk_text.pack(recs, 100)
        self.assertEqual([r for c in chunks for r in c], recs)
        self.assertTrue(all(len(c) > 0 for c in chunks))

    def test_end_to_end_round_trip(self):
        d = tempfile.mkdtemp()
        src = os.path.join(d, "src.txt")
        recs = [(b"story %d " % i) * 20 for i in range(50)]
        with open(src, "wb") as fh:
            fh.write(SEP.join(recs))
        out = os.path.join(d, "corpus")
        self.assertEqual(chunk_text.main([src, out, "--chunk-kb", "1"]), 0)
        files = sorted(os.listdir(out))
        self.assertGreater(len(files), 1)
        back = []
        for f in files:
            with open(os.path.join(out, f), "rb") as fh:
                back += chunk_text.records(fh.read(), SEP)
        self.assertEqual(back, recs)

    def test_plain_text_without_the_marker_splits_on_lines(self):
        d = tempfile.mkdtemp()
        src = os.path.join(d, "plain.txt")
        lines = [b"sentence %d about something" % i for i in range(400)]
        with open(src, "wb") as fh:
            fh.write(b"\n".join(lines))
        out = os.path.join(d, "corpus")
        self.assertEqual(
            chunk_text.main([src, out, "--chunk-kb", "1"]), 0)
        files = sorted(os.listdir(out))
        self.assertGreater(len(files), 1)   # not one big file
        back = []
        for f in files:
            with open(os.path.join(out, f), "rb") as fh:
                back += chunk_text.records(fh.read(), b"\n")
        self.assertEqual(back, lines)


if __name__ == "__main__":
    unittest.main()
