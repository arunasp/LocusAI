#!/usr/bin/env python3
"""Pack and unpack a file set as a single UTF-8 text payload.

The Filesystem MCP connector writes UTF-8 only -- its writeFileContent()
hard-codes the encoding on both its write paths -- so a tar.gz cannot be
transferred through it directly. Base64 over that text channel can, and
compresses a source tree roughly threefold on the way.

The digest is not decoration. A truncated or corrupted base64 blob decodes
into plausible garbage and fails silently at whatever reads the files next;
verifying before extraction turns that back into a loud, immediate failure.

Usage:
    transfer.py pack   OUT.b64 FILE [FILE ...]
    transfer.py unpack IN.b64 [--dest DIR] [--keep]

Stdlib only, no third-party dependencies, so it runs anywhere a bare
python3 does -- including a cicd_runner worker, whose allowlist has python3
but neither tar nor base64 as entry binaries.
"""

import argparse
import base64
import hashlib
import io
import os
import sys
import tarfile

MAGIC = "#cicd-transfer"
VERSION = 1


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def pack(out_path, files):
    """Write files as one base64 text payload with a checksummed header."""
    missing = [f for f in files if not os.path.isfile(f)]
    if missing:
        raise SystemExit("not a file: %s" % ", ".join(missing))

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for path in files:
            tar.add(path, arcname=os.path.normpath(path))
    raw = buf.getvalue()

    header = "%s v%d sha256=%s files=%d bytes=%d" % (
        MAGIC, VERSION, _digest(raw), len(files), len(raw),
    )
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(header + "\n")
        handle.write(base64.b64encode(raw).decode("ascii") + "\n")

    plain = sum(os.path.getsize(f) for f in files)
    encoded = os.path.getsize(out_path)
    print("packed %d files: %d -> %d bytes (%.2fx smaller)"
          % (len(files), plain, encoded, plain / encoded))
    return 0


def _parse_header(line):
    parts = line.split()
    if len(parts) < 2 or parts[0] != MAGIC:
        raise SystemExit("not a transfer payload: bad magic")
    if parts[1] != "v%d" % VERSION:
        raise SystemExit("unsupported payload version: %s" % parts[1])
    fields = dict(p.split("=", 1) for p in parts[2:] if "=" in p)
    if "sha256" not in fields:
        raise SystemExit("payload header carries no digest")
    return fields


def _safe_members(tar, dest):
    """Reject absolute paths, traversal, links -- extraction is not trusted."""
    root = os.path.realpath(dest)
    for member in tar.getmembers():
        if member.issym() or member.islnk():
            raise SystemExit("refusing link member: %s" % member.name)
        target = os.path.realpath(os.path.join(dest, member.name))
        if target != root and not target.startswith(root + os.sep):
            raise SystemExit("refusing path outside dest: %s" % member.name)
        yield member


def unpack(in_path, dest=".", keep=False):
    """Verify the digest, then extract. Never extracts unverified bytes."""
    with open(in_path, "r", encoding="utf-8") as handle:
        header = handle.readline().strip()
        # validate=True treats any non-alphabet byte as excess data, and the
        # payload is written with a trailing newline, so strip first.
        body = "".join(handle.read().split())

    fields = _parse_header(header)
    try:
        raw = base64.b64decode(body, validate=True)
    except Exception as exc:
        raise SystemExit("payload is not valid base64: %s" % exc)

    actual = _digest(raw)
    if actual != fields["sha256"]:
        raise SystemExit(
            "DIGEST MISMATCH -- payload truncated or corrupted\n"
            "  expected %s\n  actual   %s" % (fields["sha256"], actual)
        )
    if "bytes" in fields and len(raw) != int(fields["bytes"]):
        raise SystemExit("length mismatch despite matching digest")

    os.makedirs(dest, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
        names = [m.name for m in _safe_members(tar, dest)]
        tar.extractall(dest, members=_safe_members(tar, dest))

    print("verified sha256=%s" % actual[:16])
    for name in names:
        print("  %s" % name)
    print("unpacked %d files into %s" % (len(names), os.path.abspath(dest)))

    if not keep:
        os.remove(in_path)
        print("removed payload %s" % in_path)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="mode", required=True)

    p = sub.add_parser("pack", help="write files as one base64 payload")
    p.add_argument("out")
    p.add_argument("files", nargs="+")

    u = sub.add_parser("unpack", help="verify and extract a payload")
    u.add_argument("payload")
    u.add_argument("--dest", default=".")
    u.add_argument("--keep", action="store_true",
                   help="leave the payload in place after extracting")

    args = parser.parse_args(argv)
    if args.mode == "pack":
        return pack(args.out, args.files)
    return unpack(args.payload, args.dest, args.keep)


if __name__ == "__main__":
    sys.exit(main())
