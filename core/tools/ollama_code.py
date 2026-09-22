"""The code that serves each local Ollama model, at the version running.

  ollama_code.py OUTDIR [--api URL]

From the running server (default http://172.17.0.1:11434, the Docker
bridge gateway as seen from a container): its version, and for every
model its architecture (model_info general.architecture) and the
RENDERER / PARSER named in its modelfile. Then, blob-less and sparse:

- ollama at tag v<version> into OUTDIR/ollama-v<version>: LLAMA_CPP_VERSION,
  llama/compat/, model/parsers/ and model/renderers/. A renderer or parser
  name resolves to the file whose stem matches it (case and punctuation
  ignored), else to the non-test files that contain the quoted name --
  the dispatch registry maps some names to code in differently named
  files (qwen3.8 -> renderers/qwen35.go);
- llama.cpp at the tag LLAMA_CPP_VERSION pins into
  OUTDIR/llama.cpp-<tag>: src/models/<architecture>.cpp and the shared
  llama-arch, llama-model, llama-graph and llama-hparams sources.

Each model's anatomy from /api/show (tensor names, types, shapes and the
GGUF metadata) goes to OUTDIR/models/<model>.json. OUTDIR/manifest.json
maps every model to its files; anything unmatched is listed, not
guessed. Read-only against the server.
"""

import json
import os
import re
import subprocess
import sys
import urllib.request

API = "http://172.17.0.1:11434"
OLLAMA = "https://github.com/ollama/ollama"
LLAMACPP = "https://github.com/ggml-org/llama.cpp"
SHARED = ["src/llama-arch.cpp", "src/llama-arch.h", "src/llama-model.cpp",
          "src/llama-model.h", "src/llama-graph.cpp", "src/llama-graph.h",
          "src/llama-hparams.cpp", "src/llama-hparams.h"]


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def modelfile_field(text, key):
    """The value of the first `KEY value` line in a modelfile, or None."""
    for line in text.splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2 and parts[0] == key:
            return parts[1].strip()
    return None


def match(name, tree, prefix, ext):
    """The file under `prefix` whose stem equals `name` once case and
    punctuation are dropped, or None."""
    if not name:
        return None
    want = norm(name)
    for p in sorted(tree):
        if p.startswith(prefix) and p.endswith(ext) and "/" not in \
                p[len(prefix):] and norm(p[len(prefix):-len(ext)]) == want:
            return p
    return None


def by_literal(name, root, prefix, ext):
    """Non-test files under root/prefix that contain the quoted name."""
    if not name:
        return []
    found = []
    d = os.path.join(root, prefix)
    for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        if f.endswith(ext) and not f.endswith("_test" + ext):
            with open(os.path.join(d, f), errors="replace") as fh:
                if '"%s"' % name in fh.read():
                    found.append(prefix + f)
    return found


def resolve(name, tree, root, prefix, ext):
    """[files] implementing `name`: a stem match, else literal users."""
    m = match(name, tree, prefix, ext)
    return [m] if m else by_literal(name, root, prefix, ext)


def show(files, name):
    """Short form of resolved files for the report."""
    if not name:
        return "-"
    return ",".join(f.split("/")[-1] for f in files) or "UNMATCHED " + name


def api(base, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, data=data, headers={
        "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def sparse(url, ref, dest, paths):
    """Blob-less clone of `ref` into dest; check out `paths` only.
    Returns (commit, set of every path in the tree)."""
    if not os.path.isdir(os.path.join(dest, ".git")):
        subprocess.run(["git", "clone", "-q", "--depth", "1", "--branch",
                        ref, "--filter=blob:none", "--no-checkout", url,
                        dest], check=True)
    g = ["git", "-C", dest]
    tree = set(subprocess.run(g + ["ls-tree", "-r", "HEAD", "--name-only"],
                              check=True, capture_output=True,
                              text=True).stdout.split())
    want = [p for p in paths if p in tree or p.endswith("/")]
    if want:
        subprocess.run(g + ["sparse-checkout", "set", "--no-cone"] + want,
                       check=True)
        subprocess.run(g + ["checkout", "-q"], check=True)
    commit = subprocess.run(g + ["rev-parse", "HEAD"], check=True,
                            capture_output=True, text=True).stdout.strip()
    return commit, tree


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    out, base = argv[0], API
    if len(argv) == 3 and argv[1] == "--api":
        base = argv[2].rstrip("/")
    os.makedirs(os.path.join(out, "models"), exist_ok=True)
    version = api(base, "/api/version")["version"]
    models = []
    for m in sorted(x["name"] for x in api(base, "/api/tags")["models"]):
        s = api(base, "/api/show", {"model": m})
        arch = s.get("model_info", {}).get("general.architecture")
        mf = s.get("modelfile", "")
        models.append({"model": m, "architecture": arch,
                       "renderer": modelfile_field(mf, "RENDERER"),
                       "parser": modelfile_field(mf, "PARSER")})
        anat = {"model": m, "details": s.get("details"),
                "model_info": s.get("model_info"),
                "tensors": s.get("tensors")}
        with open(os.path.join(out, "models", norm(m) + ".json"), "w") as fh:
            json.dump(anat, fh)
    odir = os.path.join(out, "ollama-v%s" % version)
    ocommit, otree = sparse(OLLAMA, "v%s" % version, odir, [
        "LLAMA_CPP_VERSION", "llama/compat/", "model/parsers/",
        "model/renderers/"])
    for m in models:
        m["parser_files"] = resolve(m["parser"], otree, odir,
                                    "model/parsers/", ".go")
        m["renderer_files"] = resolve(m["renderer"], otree, odir,
                                      "model/renderers/", ".go")
    with open(os.path.join(odir, "LLAMA_CPP_VERSION")) as fh:
        pin = fh.read().strip()
    ldir = os.path.join(out, "llama.cpp-%s" % pin)
    lcommit, ltree = sparse(LLAMACPP, pin, ldir, [])
    for m in models:
        m["llama_cpp_file"] = match(m["architecture"], ltree, "src/models/",
                                    ".cpp")
    lpaths = SHARED + sorted({m["llama_cpp_file"] for m in models
                              if m["llama_cpp_file"]})
    sparse(LLAMACPP, pin, ldir, lpaths)
    manifest = {"api": base, "ollama_version": version,
                "ollama_commit": ocommit, "llama_cpp_pin": pin,
                "llama_cpp_commit": lcommit, "models": models}
    with open(os.path.join(out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    print("ollama %s (%s), llama.cpp %s (%s)" % (
        version, ocommit[:10], pin, lcommit[:10]))
    for m in models:
        print("  %-34s %-15s %-30s parser %-18s renderer %s" % (
            m["model"], m["architecture"],
            m["llama_cpp_file"] or "UNMATCHED",
            show(m["parser_files"], m["parser"]),
            show(m["renderer_files"], m["renderer"])))
    for d in (odir, ldir, os.path.join(out, "models")):
        files = [os.path.join(r, f) for r, _, fs in os.walk(d)
                 if ".git" not in r for f in fs]
        print("wrote %s  %d files, %d bytes" % (
            d, len(files), sum(os.path.getsize(f) for f in files)))
    print("wrote %s" % os.path.join(out, "manifest.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
