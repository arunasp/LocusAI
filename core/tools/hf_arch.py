"""Fetch the transformers source for every architecture already fetched.

  hf_arch.py HFDIR [--ref REF]

Reads every HFDIR/<org>/<name>/config.json written by hf_code.py, collects
their `model_type`s, and sparse-clones only
src/transformers/models/<model_type>/ from huggingface/transformers
(blob-less, depth 1, at REF, default the default branch) into
HFDIR/transformers. Architectures absent from transformers are reported,
not guessed: their code, if any, is in their own repository.
HFDIR/transformers/manifest.json records the commit, the architectures
found and missing, and the files checked out.
"""

import glob
import json
import os
import subprocess
import sys

REPO = "https://github.com/huggingface/transformers"


def model_types(hfdir):
    """{model_type: [repo, ...]} from HFDIR/*/*/config.json."""
    out = {}
    for p in sorted(glob.glob(os.path.join(hfdir, "*", "*", "config.json"))):
        repo = os.path.relpath(os.path.dirname(p), hfdir)
        if repo.startswith("transformers"):
            continue
        with open(p) as fh:
            t = json.load(fh).get("model_type")
        if t:
            out.setdefault(t, []).append(repo)
    return out


def git(dest, *args):
    return subprocess.run(["git", "-C", dest] + list(args), check=True,
                          capture_output=True, text=True).stdout


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    hfdir, ref = argv[0], None
    if len(argv) == 3 and argv[1] == "--ref":
        ref = argv[2]
    types = model_types(hfdir)
    dest = os.path.join(hfdir, "transformers")
    if not os.path.isdir(os.path.join(dest, ".git")):
        cmd = ["git", "clone", "-q", "--depth", "1", "--filter=blob:none",
               "--no-checkout"] + (["--branch", ref] if ref else []) + [
            REPO, dest]
        subprocess.run(cmd, check=True)
    tree = set(git(dest, "ls-tree", "-r", "HEAD", "--name-only").split())
    found, missing = {}, {}
    for t, repos in sorted(types.items()):
        d = "src/transformers/models/%s/" % t
        (found if any(p.startswith(d) for p in tree) else missing)[t] = repos
    paths = ["src/transformers/models/%s/" % t for t in found]
    if paths:
        git(dest, "sparse-checkout", "set", "--no-cone", *paths)
        git(dest, "checkout", "-q")
    commit = git(dest, "rev-parse", "HEAD").strip()
    files = sorted(p for p in tree
                   if any(p.startswith(x) for x in paths))
    manifest = {"repo": REPO, "commit": commit, "found": found,
                "missing": missing, "files": files}
    with open(os.path.join(dest, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    print("transformers %s" % commit[:12])
    for t, repos in sorted(found.items()):
        n = sum(1 for f in files if f.startswith(
            "src/transformers/models/%s/" % t))
        print("  found    %-14s %2d files  for %s" % (t, n, ", ".join(repos)))
    for t, repos in sorted(missing.items()):
        print("  missing  %-14s (code, if any, is in %s)" % (
            t, ", ".join(repos)))
    size = sum(os.path.getsize(os.path.join(dest, f)) for f in files
               if os.path.exists(os.path.join(dest, f)))
    print("wrote %s/ (%d files, %d bytes) and %s" % (
        dest, len(files), size, os.path.join(dest, "manifest.json")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
