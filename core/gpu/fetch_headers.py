#!/usr/bin/env python3
"""Fetch and extract the HIP SDK headers into core/gpu/.rocm-include.

WHY THIS EXISTS RATHER THAN AN apt INSTALL
The host's ROCm 7.14 package set ships hipcc, amdclang++ and LLVM but no
SDK headers, so `#include <hip/hip_runtime.h>` cannot resolve and no env
var or mount fixes it. The package that provides them is
`amdrocm-runtime-dev7.14` (494 KB, 179 files), but installing it via apt
pulls `amdrocm-llvm-dev7.14` as a hard dependency: 140.7 MiB downloaded
and 1976 MiB installed. Extracting just the header tree from the 0.5 MiB
.deb avoids that entirely.

The trade, stated rather than hidden: unpacked files are not
package-tracked, so a later ROCm move leaves them stale and
version-skewed. That is exactly what `make hip-header-check` exists to
catch -- it compares this tree's hip_version.h against the host's
share/hip/version and fails the build loudly. Regenerate with this
script rather than restoring from anywhere.

WHY NOT A CONTAINER IMAGE INSTEAD
Checked against Docker Hub's rocm org: AMD publishes no thin dev image at
7.x. The 6.4 line has a plain tag at 0.96 GiB, but 7.14 and 10.0 are
"-full" only at 7.40-7.70 GiB compressed. The thin images that exist are
ROCm 6.4, and 6.4 headers against a 7.14 runtime is precisely the skew
that compiles cleanly and reports wrong struct offsets. So this 494 KB
extraction is the only source that is both thin and version-exact,
because it comes from the same package set the host actually runs.

Enumerates every path it writes, with sizes, so the log states what
landed rather than leaving the reader to infer it.
"""

import gzip
import io
import lzma
import os
import sys
import tarfile
import urllib.request

REPO = "https://repo.radeon.com/rocmradeon/apt/26.13/"
DEB = ("pool/main/a/amdrocm-runtime-dev7.14/"
       "amdrocm-runtime-dev7.14_7.14.0~pre3-29052710811_amd64.deb")
PREFIX = "./opt/rocm/core-7.14/include/"
DEST = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    ".rocm-include")


def ar_members(blob):
    """Minimal ar reader. A .deb is an ar archive of three members; only
    data.tar.* is wanted and pulling in a dependency to read it would be
    absurd for a 60-byte header format."""
    if blob[:8] != b"!<arch>\n":
        raise ValueError("not an ar archive -- did the URL return HTML?")
    off = 8
    while off + 60 <= len(blob):
        hdr = blob[off:off + 60]
        name = hdr[0:16].decode().strip().rstrip("/")
        size = int(hdr[48:58].decode().strip())
        off += 60
        yield name, blob[off:off + size]
        off += size + (size % 2)


def main():
    url = REPO + DEB
    print("fetching %s" % url)
    blob = urllib.request.urlopen(url, timeout=180).read()
    print("deb bytes: %d" % len(blob))

    data = None
    for name, body in ar_members(blob):
        print("  member %s (%d bytes)" % (name, len(body)))
        if name.startswith("data.tar"):
            data = (name, body)
    if data is None:
        print("no data.tar member found", file=sys.stderr)
        return 1

    name, body = data
    if name.endswith(".xz"):
        raw = lzma.decompress(body)
    elif name.endswith(".gz"):
        raw = gzip.decompress(body)
    else:
        print("unsupported compression: %s" % name, file=sys.stderr)
        return 1

    tar = tarfile.open(fileobj=io.BytesIO(raw))
    # Member-by-member rather than extractall: nothing outside PREFIX can
    # land anywhere, no absolute paths, no links, and no dependence on
    # the extraction filter whose default changes across Python versions.
    nf = nd = skipped = 0
    for m in tar.getmembers():
        if not m.name.startswith(PREFIX):
            skipped += 1
            continue
        rel = m.name[len(PREFIX):]
        if not rel or rel.startswith("/") or ".." in rel.split("/"):
            skipped += 1
            continue
        out = os.path.join(DEST, rel)
        if m.isdir():
            os.makedirs(out, exist_ok=True)
            nd += 1
        elif m.isreg():
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with open(out, "wb") as fh:
                fh.write(tar.extractfile(m).read())
            nf += 1
        else:
            skipped += 1
    print("extracted %d files, %d dirs, skipped %d members"
          % (nf, nd, skipped))

    total = 0
    for root, _, files in os.walk(DEST):
        for fn in files:
            total += os.path.getsize(os.path.join(root, fn))
    print("[artifact] %s  %d bytes in %d files" % (DEST, total, nf))
    hv = os.path.join(DEST, "hip", "hip_version.h")
    if os.path.isfile(hv):
        print("[artifact] %s  %d bytes" % (hv, os.path.getsize(hv)))
    else:
        print("hip/hip_version.h MISSING -- hip-header-check will fail",
              file=sys.stderr)
        return 1
    print("now run: make hip-header-check")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
