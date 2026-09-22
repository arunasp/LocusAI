"""ollama_code: modelfile fields and source-file matching."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

import ollama_code as oc  # noqa: E402

TREE = {"model/parsers/qwen35.go", "model/parsers/qwen3coder.go",
        "model/parsers/nemotron3nano.go", "model/parsers/sub/qwen35.go",
        "src/models/nemotron-h-moe.cpp", "src/models/nemotron-h.cpp",
        "src/models/qwen3moe.cpp", "src/models/qwen35.cpp"}


class TestOllamaCode(unittest.TestCase):

    def test_modelfile_field(self):
        mf = "FROM /x\nRENDERER qwen3.8\nPARSER qwen3.5\nPARAMETER a 1\n"
        self.assertEqual(oc.modelfile_field(mf, "RENDERER"), "qwen3.8")
        self.assertEqual(oc.modelfile_field(mf, "PARSER"), "qwen3.5")
        self.assertIsNone(oc.modelfile_field(mf, "SYSTEM"))

    def test_match_ignores_case_and_punctuation(self):
        self.assertEqual(oc.match("qwen3.5", TREE, "model/parsers/", ".go"),
                         "model/parsers/qwen35.go")
        self.assertEqual(oc.match("nemotron-3-nano", TREE, "model/parsers/",
                                  ".go"), "model/parsers/nemotron3nano.go")
        self.assertEqual(oc.match("nemotron_h_moe", TREE, "src/models/",
                                  ".cpp"), "src/models/nemotron-h-moe.cpp")

    def test_match_is_exact_not_prefix_and_not_nested(self):
        self.assertIsNone(oc.match("nemotron_h_mo", TREE, "src/models/",
                                   ".cpp"))
        self.assertEqual(oc.match("nemotron_h", TREE, "src/models/", ".cpp"),
                         "src/models/nemotron-h.cpp")
        self.assertIsNone(oc.match(None, TREE, "src/models/", ".cpp"))


class TestResolve(unittest.TestCase):

    def test_registry_name_resolves_to_the_file_using_it(self):
        import tempfile
        root = tempfile.mkdtemp()
        d = os.path.join(root, "model", "renderers")
        os.makedirs(d)
        files = {"renderer.go": 'case "qwen3.8":\n',
                 "qwen35.go": 'slog.Warn("x", "renderer", "qwen3.8")\n',
                 "qwen38_test.go": '"qwen3.8"\n',
                 "gemma4.go": "package renderers\n"}
        for f, text in files.items():
            with open(os.path.join(d, f), "w") as fh:
                fh.write(text)
        tree = {"model/renderers/" + f for f in files}
        self.assertEqual(
            oc.resolve("qwen3.8", tree, root, "model/renderers/", ".go"),
            ["model/renderers/qwen35.go", "model/renderers/renderer.go"])
        self.assertEqual(
            oc.resolve("gemma4", tree, root, "model/renderers/", ".go"),
            ["model/renderers/gemma4.go"])
        self.assertEqual(
            oc.resolve(None, tree, root, "model/renderers/", ".go"), [])


if __name__ == "__main__":
    unittest.main()
