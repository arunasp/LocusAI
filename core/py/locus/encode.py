"""Byte-stream sensory encoding: byte units plus hashed n-gram units.

Input arrives as a byte stream (CONSTITUTION.md: streams, not imports).
At each position the active input set is the unit of the byte just read
and, for every n-gram order and hash head, the unit of the n-gram ending
there, following the Engram lookup design.

Table sizes are derived from the stream they encode, not chosen: per
order, the smallest prime at or above the number of distinct n-grams of
that order, with a distinct prime per head. Collisions are measured by
`collisions()`, never assumed away.
"""

import os
import zlib

BYTE_UNITS = 256
SKIP_DIRS = frozenset({".git", ".venv", "build", "__pycache__",
                       "node_modules", ".rocm-include", "wheels", "logs"})


def is_prime(x):
    if x < 2:
        return False
    if x % 2 == 0:
        return x == 2
    f = 3
    while f * f <= x:
        if x % f == 0:
            return False
        f += 2
    return True


def next_prime(x):
    """Smallest prime >= x."""
    x = max(x, 2)
    while not is_prime(x):
        x += 1
    return x


def repo_files(root):
    """(relative path, bytes) for every text file under root, in sorted
    path order. A file containing a NUL byte is treated as binary."""
    out = []
    for d, dirs, files in os.walk(root):
        dirs[:] = sorted(x for x in dirs if x not in SKIP_DIRS)
        for f in sorted(files):
            p = os.path.join(d, f)
            try:
                with open(p, "rb") as fh:
                    data = fh.read()
            except OSError:
                continue
            if b"\0" in data or not data:
                continue
            out.append((os.path.relpath(p, root), data))
    return out


def _fnv1a(data, seed):
    h = (2166136261 ^ seed) & 0xFFFFFFFF
    for byte in data:
        h ^= byte
        h = (h * 16777619) & 0xFFFFFFFF
    return h


class NgramEncoder:
    """Maps a byte stream to active unit sets.

    `orders` and `heads` are the design's free parameters; table sizes
    follow from `sample`, the streams the encoder is sized on.
    """

    def __init__(self, sample, orders=(2, 3, 4), heads=2):
        if heads < 1:
            raise ValueError("heads must be >= 1")
        self.orders = tuple(orders)
        self.heads = heads
        self.distinct = {}
        for k in self.orders:
            seen = set()
            for data in sample:
                for i in range(len(data) - k + 1):
                    seen.add(bytes(data[i:i + k]))
            self.distinct[k] = len(seen)
        self.tables = []            # (order, head, seed, offset, size)
        offset = BYTE_UNITS
        for k in self.orders:
            size = self.distinct[k]
            for h in range(self.heads):
                size = next_prime(size + (1 if h else 0))
                self.tables.append((k, h, (k << 8) | h, offset, size))
                offset += size
        self.n = offset

    def units_at(self, data, pos):
        """Active units after reading data[pos]."""
        out = [data[pos]]
        for k, _h, seed, offset, size in self.tables:
            if pos >= k - 1:
                gram = bytes(data[pos - k + 1:pos + 1])
                out.append(offset + _fnv1a(gram, seed) % size)
        return out

    def collisions(self, sample):
        """Per table: (order, head, size, distinct n-grams, n-grams that
        share their slot with another n-gram)."""
        report = []
        for k, h, seed, _offset, size in self.tables:
            grams = set()
            for data in sample:
                for i in range(len(data) - k + 1):
                    grams.add(bytes(data[i:i + k]))
            slots = {}
            for g in grams:
                slots.setdefault(_fnv1a(g, seed) % size, []).append(g)
            shared = sum(len(v) for v in slots.values() if len(v) > 1)
            report.append((k, h, size, len(grams), shared))
        return report

    def joint_collisions(self, sample):
        """Per order: (order, distinct n-grams, n-grams whose slots in
        every head coincide with another n-gram's). Only these are
        indistinguishable to the encoder."""
        report = []
        for k in self.orders:
            tabs = [t for t in self.tables if t[0] == k]
            grams = set()
            for data in sample:
                for i in range(len(data) - k + 1):
                    grams.add(bytes(data[i:i + k]))
            keys = {}
            for g in grams:
                key = tuple(_fnv1a(g, seed) % size
                            for _k, _h, seed, _o, size in tabs)
                keys.setdefault(key, []).append(g)
            joint = sum(len(v) for v in keys.values() if len(v) > 1)
            report.append((k, len(grams), joint))
        return report


def held_out(path, every):
    """True for roughly 1 in `every` paths, stable across runs."""
    return zlib.crc32(path.encode()) % every == 0
