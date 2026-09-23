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


class ReplayAppendsExactly(unittest.TestCase):
    """The replay draw is one RNG stream consumed file by file, so an
    appended file draws AFTER every earlier draw and carrying the state
    forward must reproduce a full rebuild exactly.

    This is the case that matters: the learner the ablations use has
    replay on, so the non-replay path above is never taken by them.
    """

    def test_appending_under_replay_equals_rebuilding(self):
        got = G.with_extra(G.prepare(FILES, replay=True), EXTRA)
        main, online = G.schedule(FILES + [EXTRA], True)
        self.assertEqual(list(got["main"]), list(main))
        self.assertEqual(list(got["online"]), list(online))

    def test_the_replay_batch_is_actually_there(self):
        # Without it the arrays would match a NON-replay rebuild, and
        # the test above would pass while proving nothing.
        got = G.with_extra(G.prepare(FILES, replay=True), EXTRA)
        plain, _ = G.schedule(FILES + [EXTRA], False)
        self.assertNotEqual(list(got["main"]), list(plain))
        self.assertGreater(len(got["main"]), len(plain))

    def test_a_different_appended_file_gives_a_different_schedule(self):
        base = G.prepare(FILES, replay=True)
        one = G.with_extra(base, EXTRA)
        two = G.with_extra(base, EXTRA + b"and more text here\n")
        self.assertNotEqual(list(one["main"]), list(two["main"]))

    def test_replay_really_does_differ(self):
        plain, _ = G.schedule(FILES, False)
        replay, _ = G.schedule(FILES, True)
        self.assertNotEqual(list(plain), list(replay))


if __name__ == "__main__":
    unittest.main()
