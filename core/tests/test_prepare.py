"""A prepared train set must equal a rebuilt one, exactly.

`exp_learn_gpu.prepare()` and `with_extra()` let an ablation reuse the
concatenated bytes and the position arrays across arms instead of
rebuilding them per run. That is only worth having if the result is
IDENTICAL to what device_run would have built itself -- a schedule that
is merely similar trains a different model and the comparison the
ablation exists for is void.

So these compare against a full rebuild, element by element, rather
than checking that the code looks right.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import exp_learn_gpu as G  # noqa: E402


FILES = [b"once upon a time there was a cat\n",
         b"the cat sat on the mat and slept\n",
         b"a dog barked at the cat\n"]
EXTRA = b"generated text appended as one more file\n"


class PreparedEqualsRebuilt(unittest.TestCase):

    def test_prepare_matches_blob_and_schedule(self):
        base = G.prepare(FILES)
        tr, tfs = G.blob(FILES)
        main, online = G.schedule(FILES, False)
        self.assertEqual(base["tr"], tr)
        self.assertEqual(list(base["tfs"]), list(tfs))
        self.assertEqual(list(base["main"]), list(main))
        self.assertEqual(list(base["online"]), list(online))
        self.assertEqual(base["n"], len(FILES))

    def test_appending_equals_rebuilding_with_the_file(self):
        # The claim the optimisation rests on: a file added at the end
        # adds its own range and changes nothing before it.
        got = G.with_extra(G.prepare(FILES), EXTRA)
        full = FILES + [EXTRA]
        tr, tfs = G.blob(full)
        main, online = G.schedule(full, False)
        self.assertEqual(got["tr"], tr)
        self.assertEqual(list(got["tfs"]), list(tfs))
        self.assertEqual(list(got["main"]), list(main))
        self.assertEqual(list(got["online"]), list(online))
        self.assertEqual(got["n"], len(full))

    def test_appending_twice_is_independent(self):
        # Each arm appends its own file to the SAME base, so the base
        # must not be mutated by the first append.
        base = G.prepare(FILES)
        first = G.with_extra(base, EXTRA)
        second = G.with_extra(base, b"a different file\n")
        self.assertEqual(list(base["main"]),
                         list(G.schedule(FILES, False)[0]))
        self.assertNotEqual(first["tr"], second["tr"])
        self.assertEqual(len(first["main"]), len(base["main"])
                         + len(EXTRA) - 1)


class ReplayIsRefused(unittest.TestCase):
    """With replay a file's batch is drawn against the episodes before
    it, so appending changes draws that are already in the array."""

    def test_with_extra_refuses_a_replay_base(self):
        base = G.prepare(FILES, replay=True)
        with self.assertRaises(ValueError):
            G.with_extra(base, EXTRA)

    def test_replay_really_does_differ(self):
        # If this ever stops being true the refusal above is pointless.
        plain, _ = G.schedule(FILES, False)
        replay, _ = G.schedule(FILES, True)
        self.assertNotEqual(list(plain), list(replay))


if __name__ == "__main__":
    unittest.main()
