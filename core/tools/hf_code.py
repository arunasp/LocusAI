"""Fetch a Hugging Face model repository without its tensors.

  hf_code.py REPO DEST [--revision R] [--max-mb N] [--dry-run]

Lists REPO (e.g. deepseek-ai/DeepSeek-V4.1-Flash) through the Hub API and
downloads every file except tensor and weight files (by extension) and
any file larger than N MB (default 50). Paths are kept under DEST/REPO.
Each downloaded file is checked against the Hub's hash: the git blob
SHA-1 for ordinary files, the SHA-256 for LFS files. DEST/REPO/
manifest.json lists every file kept and every file skipped, with its size
and the reason, so the tensor layout stays known without the tensors.

--dry-run lists and filters only; nothing is downloaded. HF_TOKEN, if
set, is sent as a bearer token (gated or private repositories).
Standard library only.
"""

import hashlib
import json
import os
import re
import sys
import urllib.parse
import urllib.request

HUB = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
TENSOR_EXT = (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf",
              ".ggml", ".h5", ".msgpack", ".onnx", ".npz", ".npy", ".pkl",
              ".tflite", ".mlmodel", ".ot")


def _request(url):
    req = urllib.request.Request(url, headers={"User-Agent": "hf_code.py"})
    token = os.environ.get("HF_TOKEN")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    return urllib.request.urlopen(req, timeout=60)


def list_tree(repo, revision):
    """Every file entry in the repository, following Link pagination."""
    url = "%s/api/models/%s/tree/%s?recursive=true" % (
        HUB, repo, urllib.parse.quote(revision, safe=""))
    files = []
    while url:
        with _request(url) as r:
            files += [e for e in json.load(r) if e.get("type") == "file"]
            m = re.search(r'<([^>]+)>;\s*rel="next"',
                          r.headers.get("Link", ""))
        url = m.group(1) if m else None
    return files


def size_of(entry):
    return (entry.get("lfs") or {}).get("size", entry.get("size", 0))


def skip_reason(entry, max_bytes):
    path = entry["path"].lower()
    if path.endswith(TENSOR_EXT):
        return "tensor file"
    if size_of(entry) > max_bytes:
        return "over size cap"
    return None


def verify(entry, data):
    """True when data matches the Hub's recorded hash for this entry."""
    lfs = entry.get("lfs")
    if lfs and lfs.get("oid"):
        return hashlib.sha256(data).hexdigest() == lfs["oid"]
    blob = b"blob %d\0" % len(data) + data
    return hashlib.sha1(blob).hexdigest() == entry.get("oid")


def fetch(repo, dest, revision="main", max_mb=50.0, dry_run=False):
    """Returns the manifest dict; raises on a hash mismatch."""
    root = os.path.join(dest, repo)
    max_bytes = int(max_mb * 1024 * 1024)
    kept, skipped = [], []
    for e in list_tree(repo, revision):
        why = skip_reason(e, max_bytes)
        rec = {"path": e["path"], "size": size_of(e)}
        if why:
            skipped.append(dict(rec, reason=why))
            continue
        kept.append(rec)
        if dry_run:
            continue
        url = "%s/%s/resolve/%s/%s" % (
            HUB, repo, urllib.parse.quote(revision, safe=""),
            urllib.parse.quote(e["path"]))
        with _request(url) as r:
            data = r.read()
        if not verify(e, data):
            raise ValueError("hash mismatch: %s" % e["path"])
        out = os.path.join(root, e["path"])
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "wb") as fh:
            fh.write(data)
    manifest = {"repo": repo, "revision": revision, "max_mb": max_mb,
                "dry_run": dry_run, "kept": kept, "skipped": skipped}
    if not dry_run:
        os.makedirs(root, exist_ok=True)
        with open(os.path.join(root, "manifest.json"), "w") as fh:
            json.dump(manifest, fh, indent=1)
    return manifest


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    repo, dest, rest = argv[0], argv[1], argv[2:]
    revision, max_mb, dry = "main", 50.0, False
    while rest:
        if rest[0] == "--revision" and len(rest) > 1:
            revision, rest = rest[1], rest[2:]
        elif rest[0] == "--max-mb" and len(rest) > 1:
            max_mb, rest = float(rest[1]), rest[2:]
        elif rest[0] == "--dry-run":
            dry, rest = True, rest[1:]
        else:
            print("unknown argument: %s" % rest[0])
            return 2
    m = fetch(repo, dest, revision, max_mb, dry)
    ks = sum(f["size"] for f in m["kept"])
    ss = sum(f["size"] for f in m["skipped"])
    print("%s@%s: keep %d files (%.1f MB), skip %d files (%.1f GB)%s" % (
        repo, revision, len(m["kept"]), ks / 2**20, len(m["skipped"]),
        ss / 2**30, " -- dry run, nothing downloaded" if dry else ""))
    for f in m["kept"]:
        print("  keep  %10d  %s" % (f["size"], f["path"]))
    reasons = {}
    for f in m["skipped"]:
        reasons.setdefault(f["reason"], []).append(f)
    for why, fs in sorted(reasons.items()):
        print("  skip  %d files, %.1f GB: %s (e.g. %s)" % (
            len(fs), sum(f["size"] for f in fs) / 2**30, why, fs[0]["path"]))
    if not dry:
        root = os.path.join(dest, repo)
        print("wrote %s/ (%d files) and %s  %d bytes" % (
            root, len(m["kept"]), os.path.join(root, "manifest.json"),
            os.path.getsize(os.path.join(root, "manifest.json"))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
