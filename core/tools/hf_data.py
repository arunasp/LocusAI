"""Fetch a bounded slice of a Hugging Face dataset as reading material.

  hf_data.py REPO DEST [--file PATH ...] [--max-mb N] [--dry-run]
  hf_data.py REPO DEST --rows [--max-mb N] [--config C --split S]
                              [--column NAME]

With --rows the text is streamed through the Hub's rows API, which
serves any dataset as JSON. That is how a parquet-only corpus is read
here at all: the project has no parquet reader and no dependency
beyond the standard library, and a prefix of a parquet file is not
text. The rows are concatenated one per line into DEST/REPO/rows.txt up
to --max-mb, and the manifest records the dataset, config, split,
column, row count and the text's SHA-256.

Otherwise, lists REPO through the Hub's dataset API. Without --file it
prints the files and stops. Each named file is downloaded into DEST/REPO, up to
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
import time
import urllib.parse
import urllib.request

HUB = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
ROWS = os.environ.get("HF_ROWS_ENDPOINT",
                      "https://datasets-server.huggingface.co")
PAGE = 100                      # rows per request, the API's maximum
# One chunk is the memory ceiling for a file fetch, whatever the file's
# size; 1 MiB is large enough that the syscall count is irrelevant next
# to the transfer and small enough to be invisible in RSS.
CHUNK = 1 << 20
# How often the fetch says where it is. An initial value, not measured:
# the open question is whether progress should be reported on TIME at
# all or on bytes transferred, which is what a reader actually wants to
# compare against the published size.
PROGRESS_SECONDS = 10


def _request(url, headers=None, tries=5):
    """Open `url`, retrying a server-side failure with backoff.

    The rows API returns 502 and 503 under load; a whole corpus fetch
    was lost to one 502 mid-stream before this existed. Retries are
    bounded and only for server errors -- a 404 is an answer, not a
    hiccup.
    """
    h = {"User-Agent": "hf_data.py"}
    h.update(headers or {})
    token = os.environ.get("HF_TOKEN")
    if token:
        h["Authorization"] = "Bearer " + token
    req = urllib.request.Request(url, headers=h)
    for attempt in range(tries):
        try:
            return urllib.request.urlopen(req, timeout=120)
        except urllib.error.HTTPError as e:
            if e.code < 500 or attempt == tries - 1:
                raise
        except urllib.error.URLError:
            if attempt == tries - 1:
                raise
        time.sleep(2.0 ** attempt)


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
    out = os.path.join(dest, path)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    # COMMITTED IN CHUNKS, never accumulated whole. `r.read()` held the
    # entire slice in memory and wrote once at the end, so a 512 MiB
    # prefix sat as 512 MiB of RSS while the file on disk stayed at its
    # old size -- indistinguishable from a stalled fetch, and unbounded
    # in the one dimension that matters, since the published files here
    # run to 2.2 GB. Each chunk is written and hashed as it arrives, so
    # memory is one chunk and the partial download is on disk where its
    # growth can be watched.
    #
    # Written to `out + ".part"` and renamed only on success: a truncated
    # or failed transfer must never be left under the real name, where
    # the next run would read it as a complete corpus.
    digest = hashlib.sha256()
    got = 0
    part = out + ".part"
    last = time.time()
    with _request(url, headers) as r, open(part, "wb") as fh:
        while True:
            chunk = r.read(CHUNK)
            if not chunk:
                break
            fh.write(chunk)
            digest.update(chunk)
            got += len(chunk)
            now = time.time()
            if now - last >= PROGRESS_SECONDS:
                print("  %-40s %6.1f MB" % (path, got / 2**20),
                      flush=True)
                last = now
    os.replace(part, out)
    rec = {"path": path, "published_size": full, "bytes": got,
           "truncated": truncated,
           "sha256": digest.hexdigest()}
    oid = (entry.get("lfs") or {}).get("oid")
    if not truncated and oid:
        rec["verified"] = rec["sha256"] == oid
        if not rec["verified"]:
            raise ValueError("hash mismatch: %s" % path)
    return rec


def first_split(dataset):
    """(config, split) of the first split the rows API reports."""
    with _request("%s/splits?dataset=%s"
                  % (ROWS, urllib.parse.quote(dataset))) as r:
        s = json.load(r)["splits"]
    if not s:
        raise ValueError("no splits for %s" % dataset)
    return s[0]["config"], s[0]["split"]


def fetch_rows(dataset, dest, max_bytes, config=None, split=None,
               column=None):
    """Stream rows into dest/rows.txt; returns the manifest record."""
    if config is None or split is None:
        config, split = first_split(dataset)
    base = ("%s/rows?dataset=%s&config=%s&split=%s"
            % (ROWS, urllib.parse.quote(dataset),
               urllib.parse.quote(config), urllib.parse.quote(split)))
    out = os.path.join(dest, "rows.txt")
    h = hashlib.sha256()
    total = rows = 0
    offset, done, partial = 0, False, None
    with open(out, "wb") as fh:
        while not done:
            try:
                with _request("%s&offset=%d&length=%d"
                              % (base, offset, PAGE)) as r:
                    page = json.load(r)
            except Exception as e:
                # Keep what arrived: a prefix of a corpus is still a
                # stream to read, and the manifest says where it ended.
                partial = "%s at offset %d" % (type(e).__name__, offset)
                break
            if column is None:
                names = [f["name"] for f in page.get("features", [])]
                column = names[0] if names else "text"
            got = page.get("rows", [])
            if not got:
                break
            for item in got:
                value = item.get("row", {}).get(column)
                if value is None:
                    continue
                line = ("%s\n" % value).encode("utf-8", "replace")
                if total + len(line) > max_bytes:
                    done = True
                    break
                fh.write(line)
                h.update(line)
                total += len(line)
                rows += 1
            offset += len(got)
            if page.get("num_rows_total") and offset >= page[
                    "num_rows_total"]:
                break
    rec = {"path": "rows.txt", "dataset": dataset, "config": config,
           "split": split, "column": column, "rows": rows,
           "bytes": total, "sha256": h.hexdigest()}
    if partial:
        rec["partial"] = partial
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
    rows, config, split, column = False, None, None, None
    while rest:
        if rest[0] == "--rows":
            rows, rest = True, rest[1:]
        elif rest[0] == "--config" and len(rest) > 1:
            config, rest = rest[1], rest[2:]
        elif rest[0] == "--split" and len(rest) > 1:
            split, rest = rest[1], rest[2:]
        elif rest[0] == "--column" and len(rest) > 1:
            column, rest = rest[1], rest[2:]
        elif rest[0] == "--file" and len(rest) > 1:
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
    if rows:
        root = os.path.join(dest, repo)
        os.makedirs(root, exist_ok=True)
        recs, lic = [], None
        if "README.md" in by_path:
            recs.append(fetch_file(repo, by_path["README.md"], root,
                                   int(max_mb * 2**20)))
            with open(os.path.join(root, "README.md"),
                      errors="replace") as fh:
                lic = licence(fh.read())
        rec = fetch_rows(repo, root, int(max_mb * 2**20), config, split,
                         column)
        recs.append(rec)
        with open(os.path.join(root, "manifest.json"), "w") as fh:
            json.dump({"repo": repo, "license": lic, "max_mb": max_mb,
                       "mode": "rows", "files": recs}, fh, indent=1)
        print("%s, license %s" % (repo, lic or "unstated in README"))
        print("  %d rows, %d bytes from %s/%s column %s%s"
              % (rec["rows"], rec["bytes"], rec["config"], rec["split"],
                 rec["column"],
                 "\n  PARTIAL: stopped on %s" % rec["partial"]
                 if rec.get("partial") else ""))
        print("wrote %s/rows.txt and %s/manifest.json" % (root, root))
        return 0
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
