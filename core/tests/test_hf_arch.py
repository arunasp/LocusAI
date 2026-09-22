"""hf_arch.model_types: which architectures the fetched configs name."""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

import hf_arch  # noqa: E402


def write(root, repo, cfg):
    d = os.path.join(root, repo)
    os.makedirs(d)
    with open(os.path.join(d, "config.json"), "w") as fh:
        json.dump(cfg, fh)


class TestModelTypes(unittest.TestCase):

    def test_groups_repos_by_model_type(self):
        root = tempfile.mkdtemp()
        write(root, "a/one", {"model_type": "qwen2"})
        write(root, "b/two", {"model_type": "qwen2"})
        write(root, "c/three", {"model_type": "gemma4"})
        write(root, "d/untyped", {"architectures": ["X"]})
        self.assertEqual(hf_arch.model_types(root),
                         {"qwen2": ["a/one", "b/two"],
                          "gemma4": ["c/three"]})

    def test_ignores_its_own_checkout(self):
        root = tempfile.mkdtemp()
        write(root, "transformers/x", {"model_type": "qwen2"})
        self.assertEqual(hf_arch.model_types(root), {})


if __name__ == "__main__":
    unittest.main()
