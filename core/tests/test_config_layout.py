"""The ctypes mirror of LocusConfig is checked, not trusted.

A mirror that disagrees with the C struct does not raise: it reads a
neighbouring field's bytes, and the store runs on values nobody set.
These tests deliberately build wrong mirrors and require the guard to
catch each way one can drift.
"""

import ctypes
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "py"))

from locus import store as store_mod                  # noqa: E402
from locus.store import LayoutMismatch, Store         # noqa: E402


def library():
    return store_mod._bind(store_mod._default_lib_path())


def fields_of(mirror):
    return list(mirror._fields_)


class TestConfigLayoutGuard(unittest.TestCase):

    def test_the_real_mirror_matches_the_library(self):
        # _bind already checks; a Store constructing at all is the proof.
        s = Store()
        self.assertGreater(ctypes.sizeof(s.config), 0)

    def test_a_missing_field_is_caught(self):
        class Short(ctypes.Structure):
            _fields_ = fields_of(store_mod._Config)[:-1]

        with self.assertRaises(LayoutMismatch) as got:
            store_mod._check_config_layout(library(), Short)
        self.assertIn("bytes", str(got.exception))

    def test_an_added_field_is_caught(self):
        class Long(ctypes.Structure):
            _fields_ = fields_of(store_mod._Config) + [
                ("extra", ctypes.c_double)]

        with self.assertRaises(LayoutMismatch):
            store_mod._check_config_layout(library(), Long)

    def test_reordered_fields_are_caught(self):
        original = fields_of(store_mod._Config)
        swapped = list(original)
        # Swap two fields of DIFFERENT width, so the size can stay the
        # same while every later offset shifts -- the case a size check
        # alone would miss.
        i = [n for n, _ in enumerate(original) if original[n][0] == "decay"][0]
        j = [n for n, _ in enumerate(original)
             if original[n][0] == "capture_window"][0]
        swapped[i], swapped[j] = swapped[j], swapped[i]

        class Reordered(ctypes.Structure):
            _fields_ = swapped

        with self.assertRaises(LayoutMismatch):
            store_mod._check_config_layout(library(), Reordered)


if __name__ == "__main__":
    unittest.main()
