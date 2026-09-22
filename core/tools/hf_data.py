"""Fetch a bounded slice of a Hugging Face dataset as reading material.

  hf_data.py REPO DEST [--file PATH ...] [--max-mb N] [--dry-run]

Lists REPO through the Hub's dataset API. Without --file it prints the
files and stops. Each named file is downloaded into DEST/REPO, up to
--max-mb (default 32); a file larger than that is fetched as a PREFIX
with an HTTP range request and recorded as truncated, since a prefix of
a text corpus is still a stream to read. A file fetched whole is checked
against the Hub's LFS SHA-256 where one is published; a prefix cannot be,
so its own SHA-256 is recorded instead. README.md is always fetched for
the licence. DEST/REPO/manifest.json holds every file with its size,
whether it was truncated, and its hash. Standard library only.
"""

import hashlib
import json
import os
import re
import sys
import urllib.parse
import urllib.request

HUB = os.environ.get("HF_ENDPOINT", "https://huggingface.co")


def _request(url, headers=None):
    h = {"User-Agent": "hf_data.py"}
    h.update(headers or {})
    token = os.environ.get("HF_TOKEN")
    if token:
        h["Authorization"] = "Bearer " + token
    return urllib.request.urlopen(urllib.request.Request(url, headers=h),
                                  timeout=120)


def list_tree(repo, revision="main"):
    url = "%s/api/datasets/%s/tree/%s?recursive=true" % (
        HUB, repo, urllib.parse.quote(revision, safe=""))
    out = []
    while url:
        with _request(url) as r:
            out += [e for e in json.load(r) if e.get("type") == "file"]
            m = re.search(r'<([^>]+)>;\s*rel="next"',
                          r.headers.get("Link", ""))
        url = m.group(1) if m else None
    return out


def size_of(entry):
    return (entry.get("lfs") or {}).get("size", entry.get("size", 0))


def fetch_file(repo, entry, dest, max_bytes, revision="main"):
    """Returns the manifest record; downloads whole or as a prefix."""
    path = entry["path"]
    url = "%s/datasets/%s/resolve/%s/%s" % (
        HUB, repo, urllib.parse.quote(revision, safe=""),
        urllib.parse.quote(path))
    full = size_of(entry)
    truncated = full > max_bytes
    headers = {"Range": "bytes=0-%d" % (max_bytes - 1)} if truncated else {}
    with _request(url, headers) as r:
        data = r.read()
    out = os.path.join(dest, path)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "wb") as fh:
        fh.write(data)
    rec = {"path": path, "published_size": full, "bytes": len(data),
           "truncated": truncated,
           "sha256": hashlib.sha256(data).hexdigest()}
    oid = (entry.get("lfs") or {}).get("oid")
    if not truncated and oid:
        rec["verified"] = rec["sha256"] == oid
        if not rec["verified"]:
            raise ValueError("hash mismatch: %s" % path)
    return rec


def licence(text):
    """The license field of a README front-matter block, if present."""
    m = re.search(r"^---\s*$(.*?)^---\s*$", text or "", re.S | re.M)
    if not m:
        return None
    m2 = re.search(r"^license:\s*(.+)$", m.group(1), re.M)
    return m2.group(1).strip() if m2 else None


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    repo, dest, rest = argv[0], argv[1], argv[2:]
    want, max_mb, dry = [], 32.0, False
    while rest:
        if rest[0] == "--file" and len(rest) > 1:
            want.append(rest[1])
            rest = rest[2:]
        elif rest[0] == "--max-mb" and len(rest) > 1:
            max_mb, rest = float(rest[1]), rest[2:]
        elif rest[0] == "--dry-run":
            dry, rest = True, rest[1:]
        else:
            print("unknown argument: %s" % rest[0])
            return 2
    tree = list_tree(repo)
    by_path = {e["path"]: e for e in tree}
    if not want or dry:
        print("%s: %d files" % (repo, len(tree)))
        for e in sorted(tree, key=lambda x: -size_of(x))[:20]:
            print("  %14d  %s" % (size_of(e), e["path"]))
        if not want:
            print("name files with --file to fetch them")
            return 0
    missing = [p for p in want if p not in by_path]
    if missing:
        print("not in the repository: %s" % ", ".join(missing))
        return 2
    if dry:
        return 0
    root = os.path.join(dest, repo)
    os.makedirs(root, exist_ok=True)
    recs, lic = [], None
    if "README.md" in by_path:
        r = fetch_file(repo, by_path["README.md"], root, int(max_mb * 2**20))
        recs.append(r)
        with open(os.path.join(root, "README.md"), errors="replace") as fh:
            lic = licence(fh.read())
    for p in want:
        recs.append(fetch_file(repo, by_path[p], root, int(max_mb * 2**20)))
    manifest = {"repo": repo, "license": lic, "max_mb": max_mb,
                "files": recs}
    with open(os.path.join(root, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    print("%s, license %s" % (repo, lic or "unstated in README"))
    for r in recs:
        print("  %10d bytes  %s%s" % (
            r["bytes"], r["path"],
            "  (prefix of %d, truncated)" % r["published_size"]
            if r["truncated"] else
            ("  (hash verified)" if r.get("verified") else "")))
    print("wrote %s/ and %s" % (root, os.path.join(root, "manifest.json")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
