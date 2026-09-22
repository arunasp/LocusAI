"""Split one text file into corpus files at record boundaries.

  chunk_text.py SRC DEST [--sep TEXT] [--chunk-kb N]

Splits SRC on SEP (default "<|endoftext|>"), packs whole records into
files of about N kilobytes (default 64) named part-00000.txt upward
under DEST, and verifies the round trip: the records read back from the
files must equal the records read from SRC.

If SEP does not occur in SRC, records are LINES instead. Plain text
without a record marker would otherwise become a single record and so a
single file, which `exp_learn_stream.split` cannot divide into train,
validation and test -- what happened to the Simple Wikipedia rows, which
carry no marker. A corpus of many files is
what `exp_learn_stream.split` needs to hold out validation and test
material, and record boundaries keep each file whole.
"""

import os
import sys


def records(data, sep):
    return [r for r in data.split(sep) if r.strip()]


def pack(recs, chunk_bytes):
    """Records grouped into chunks of about chunk_bytes."""
    out, cur, n = [], [], 0
    for r in recs:
        cur.append(r)
        n += len(r)
        if n >= chunk_bytes:
            out.append(cur)
            cur, n = [], 0
    if cur:
        out.append(cur)
    return out


def write(chunks, dest, sep):
    os.makedirs(dest, exist_ok=True)
    paths = []
    for i, c in enumerate(chunks):
        p = os.path.join(dest, "part-%05d.txt" % i)
        with open(p, "wb") as fh:
            fh.write(sep.join(c))
        paths.append(p)
    return paths


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    src, dest, rest = argv[0], argv[1], argv[2:]
    sep, kb = b"<|endoftext|>", 64
    while rest:
        if rest[0] == "--sep" and len(rest) > 1:
            sep, rest = rest[1].encode(), rest[2:]
        elif rest[0] == "--chunk-kb" and len(rest) > 1:
            kb, rest = int(rest[1]), rest[2:]
        else:
            print("unknown argument: %s" % rest[0])
            return 2
    with open(src, "rb") as fh:
        data = fh.read()
    if sep not in data and b"\n" in data:
        print("%s: no %r -- splitting on lines instead"
              % (src, sep.decode("utf-8", "replace")))
        sep = b"\n"
    recs = records(data, sep)
    chunks = pack(recs, kb * 1024)
    paths = write(chunks, dest, sep)
    back = []
    for p in sorted(paths):
        with open(p, "rb") as fh:
            back += records(fh.read(), sep)
    if back != recs:
        print("FAIL: round trip differs: %d records in, %d back"
              % (len(recs), len(back)))
        return 1
    total = sum(os.path.getsize(p) for p in paths)
    print("%s: %d records, %d bytes" % (src, len(recs), len(data)))
    print("wrote %d files, %d bytes into %s/ (round trip verified)"
          % (len(paths), total, dest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
