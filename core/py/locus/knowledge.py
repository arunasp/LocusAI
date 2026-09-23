"""A learned knowledge store with a permanent and a transient tier.

The permanent tier is the store `core/gpu/learn_device.cpp` writes with
its KNOW argument (format in that file) plus a JSON sidecar holding the
encoder tables, the lam fitted on validation, provenance, the clock, and
the consolidation history. Only experience the learner formed is in it
(CONSTITUTION.md: streams, not imports).

Reading is learning. `Knowledge.read(data)` predicts each byte from the
units active before it, then learns the transition with the training
rule, continuing each unit's metaplastic count. The clock advances one
tick per byte read.

A changed row is tagged with the tick of its last change and held in the
transient tier (PATH.tags), which every later reader sees on top of the
permanent tier. Nothing reaches the permanent tier until sleep, which
LocusAI enters by itself: sleep pressure is the number of events learned
into the transient tier divided by the events in the permanent tier
(lifetime experience), and a read that would learn first sleeps when the
pressure has reached `sleep_threshold` (stored in the sidecar, default
0.01). At sleep:

- an outcome event (value, tick) is the modulator (`outcome`);
- at sleep each tag sums the outcomes at or after its tick and within
  `lifetime` ticks of it; a positive sum captures the row into a new
  generation of the permanent tier, anything else lets it lapse;
- the previous generation is kept as PATH.prev; the transient tier is
  cleared.

Synaptic tagging and capture (Frey & Morris 1997), eligibility traces
credited by a delayed modulator (Izhikevich 2007), offline consolidation
from a fast store into a slow one (McClelland, McNaughton & O'Reilly
1995); sleep pressure built by learning while awake (Borbely 1982;
Tononi & Cirelli 2003), falling with accumulated experience (Roffwarg,
Muzio & Dement 1966).
"""

import json
import math
import os
import time
from array import array

from .encode import BYTE_UNITS, NgramEncoder

INF = float("inf")

MAGIC = b"LOCUSKN1"
META = 1
TOP_ONLY = 2
CLIP = 5.0
SLEEP_THRESHOLD = 0.01


class Store:
    """One sparse store: per unit, its non-zero weights by next byte and
    the number of events it has learned from."""

    def __init__(self, flags, events, rowptr, byte, weight):
        self.flags = flags
        self.events = events
        self.rowptr = rowptr
        self.byte = byte
        self.weight = weight
        self.changed = {}           # tagged: unit -> {byte: weight}
        self.prior = {}             # tagged unit -> event count before
        self.tick = {}              # tagged unit -> tick of last change

    def row(self, u):
        if u in self.changed:
            return self.changed[u]
        lo, hi = self.rowptr[u], self.rowptr[u + 1]
        return dict(zip(self.byte[lo:hi], self.weight[lo:hi]))

    def learn(self, u, nxt, rate, tick):
        """One event for unit u, as `step` in learn_device.cpp."""
        row = self.row(u)
        if u not in self.prior:
            self.prior[u] = self.events[u]
        k = self.events[u] + 1
        loc = 1.0 / (rate * k) if self.flags & META else None
        for b in set(row) | {nxt}:
            w = row.get(b, 0.0)
            err = (1.0 if b == nxt else 0.0) - w
            if err == 0.0:
                continue
            dw = rate * err
            dw = dw * 1.0
            if loc is not None:
                dw = dw * loc
            if dw == 0.0:
                continue
            w = w + dw
            row[b] = CLIP if w > CLIP else (-CLIP if w < -CLIP else w)
        self.changed[u] = row
        self.events[u] = k
        self.tick[u] = tick

    def lapse(self, units):
        """Untag units: rows and event counts return to the store."""
        for u in units:
            self.events[u] = self.prior.pop(u)
            del self.changed[u]
            del self.tick[u]

    def captured(self, U):
        """(events, rowptr, byte, weight) with every tagged row captured."""
        rp, by, w = array("q", [0]), bytearray(), array("d")
        for u in range(U):
            if u in self.changed:
                row = self.changed[u]
                for b in sorted(row):
                    if row[b] != 0.0:
                        by.append(b)
                        w.append(row[b])
            else:
                lo, hi = self.rowptr[u], self.rowptr[u + 1]
                by += self.byte[lo:hi]
                w.extend(self.weight[lo:hi])
            rp.append(len(w))
        return self.events, rp, bytes(by), w


class Knowledge:

    def __init__(self, path, sleep_threshold=None):
        self.path = path
        with open(path + ".json") as fh:
            self.meta = json.load(fh)
        m = self.meta
        self.enc = NgramEncoder.from_tables(m["orders"], m["heads"],
                                            m["tables"])
        self.lam = m["lam"]
        # lam IS the floor that makes every byte possible: p(b) carries
        # lam / BYTE_UNITS of uniform mass, so at lam <= 0 an unseen
        # byte has probability zero and -log2(0) is not a number of
        # bits. A NEGATIVE lam is worse and is why this is checked
        # rather than assumed: the probabilities still SUM TO ONE, so
        # the obvious sanity check passes, while individual values go
        # negative and the bits figure comes out BETTER than physically
        # possible. It is read from a JSON sidecar on disk, so it is an
        # input, not a constant.
        if (not isinstance(self.lam, (int, float))
                or self.lam != self.lam
                or self.lam in (INF, -INF)
                or self.lam <= 0.0):
            raise ValueError("%s.json: lam must be a finite positive "
                             "number, got %r"
                             % (self.path, self.lam))
        self.gated = m.get("readout") == "gated"
        self.top = max(m["orders"])
        self.clock = m.get("clock", 0)
        self.outcomes = []          # (tick, value), transient
        self.threshold = (sleep_threshold if sleep_threshold is not None
                          else m.get("sleep_threshold", SLEEP_THRESHOLD))
        self.last_sleep = None      # (captured, lapsed) of an own sleep
        with open(path, "rb") as fh:
            raw = fh.read()
        if raw[:8] != MAGIC:
            raise ValueError("%s: not a LocusAI knowledge store" % path)
        u, ns = array("i"), array("d")
        u.frombytes(raw[8:16])
        ns.frombytes(raw[16:24])
        self.U, nstores, self.rate = u[0], u[1], ns[0]
        if self.U != self.enc.n:
            raise ValueError("store has %d units, encoder %d"
                             % (self.U, self.enc.n))
        off, self.stores = 24, []
        for _ in range(nstores):
            f = array("i")
            f.frombytes(raw[off:off + 4])
            off += 4
            ev = array("q")
            ev.frombytes(raw[off:off + 8 * self.U])
            off += 8 * self.U
            rp = array("q")
            rp.frombytes(raw[off:off + 8 * (self.U + 1)])
            off += 8 * (self.U + 1)
            nnz = rp[-1]
            by = raw[off:off + nnz]
            off += nnz
            w = array("d")
            w.frombytes(raw[off:off + 8 * nnz])
            off += 8 * nnz
            self.stores.append(Store(f[0], ev, rp, bytes(by), w))
        if off != len(raw):
            raise ValueError("%s: %d trailing bytes" % (path, len(raw) - off))
        self._count_lifetime()
        self._load_transient()

    def _count_lifetime(self):
        self.lifetime = sum(sum(s.events) for s in self.stores)

    def pressure(self):
        """Events learned into the transient tier / lifetime events."""
        new = sum(s.events[u] - s.prior[u] for s in self.stores
                  for u in s.changed)
        return new / max(self.lifetime, 1)

    # ------------------------------------------------------------ tiers --
    def _load_transient(self):
        p = self.path + ".tags"
        if not os.path.exists(p):
            return
        with open(p) as fh:
            t = json.load(fh)
        self.clock = t["clock"]
        self.outcomes = [tuple(o) for o in t["outcomes"]]
        for si, u, tick, prior, events, row in t["tags"]:
            s = self.stores[si]
            s.prior[u] = prior
            s.events[u] = events
            s.tick[u] = tick
            s.changed[u] = {int(b): w for b, w in row.items()}

    def save_transient(self):
        """Write the transient tier (tags, outcomes, clock) to PATH.tags;
        remove it when nothing is tagged and no outcome is pending."""
        p = self.path + ".tags"
        tags = [[si, u, s.tick[u], s.prior[u], s.events[u], s.changed[u]]
                for si, s in enumerate(self.stores) for u in sorted(s.changed)]
        if not tags and not self.outcomes:
            if os.path.exists(p):
                os.remove(p)
            return
        with open(p + ".tmp", "w") as fh:
            json.dump({"clock": self.clock, "outcomes": self.outcomes,
                       "tags": tags}, fh)
        os.replace(p + ".tmp", p)

    def tagged(self):
        """Number of rows in the transient tier."""
        return sum(len(s.changed) for s in self.stores)

    def outcome(self, value):
        """A modulator event at the current tick."""
        self.outcomes.append((self.clock, float(value)))

    def sleep(self, lifetime=None, note=""):
        """Offline consolidation. Each tag sums the outcomes at or after
        its tick and within `lifetime` ticks (None: no limit); a positive
        sum captures the row, anything else lets it lapse. Returns
        (captured, lapsed)."""
        lapsed = 0
        for s in self.stores:
            drop = []
            for u, t in s.tick.items():
                m = sum(v for to, v in self.outcomes
                        if to >= t and (lifetime is None
                                        or to - t <= lifetime))
                if m <= 0.0:
                    drop.append(u)
            s.lapse(drop)
            lapsed += len(drop)
        n = self.tagged()
        if n:
            self._write_generation(n, lifetime, note)
            self._count_lifetime()
        elif self.clock != self.meta.get("clock", 0):
            self.meta["clock"] = self.clock
            with open(self.path + ".json.tmp", "w") as fh:
                json.dump(self.meta, fh, indent=1)
            os.replace(self.path + ".json.tmp", self.path + ".json")
        self.outcomes = []
        for s in self.stores:
            s.changed, s.prior, s.tick = {}, {}, {}
        self.save_transient()
        return n, lapsed

    def consolidate(self, outcome, note=""):
        """An outcome at the current tick, then sleep. Returns the rows
        captured."""
        self.outcome(outcome)
        return self.sleep(note=note)[0]

    def _write_generation(self, n, lifetime, note):
        parts = [MAGIC, array("i", [self.U, len(self.stores)]).tobytes(),
                 array("d", [self.rate]).tobytes()]
        new = []
        for s in self.stores:
            ev, rp, by, w = s.captured(self.U)
            parts += [array("i", [s.flags]).tobytes(), ev.tobytes(),
                      rp.tobytes(), by, w.tobytes()]
            new.append(Store(s.flags, ev, rp, by, w))
        meta = dict(self.meta)
        meta["generation"] = meta.get("generation", 0) + 1
        meta["clock"] = self.clock
        meta["consolidations"] = meta.get("consolidations", []) + [{
            "generation": meta["generation"], "rows": n,
            "outcomes": [list(o) for o in self.outcomes],
            "lifetime": lifetime, "note": note, "clock": self.clock,
            "when": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}]
        tmp = self.path + ".tmp"
        with open(tmp, "wb") as fh:
            for part in parts:
                fh.write(part)
        with open(tmp + ".json", "w") as fh:
            json.dump(meta, fh, indent=1)
        for ext in ("", ".json"):
            if os.path.exists(self.path + ext):
                os.replace(self.path + ext, self.path + ".prev" + ext)
        os.replace(tmp, self.path)
        os.replace(tmp + ".json", self.path + ".json")
        self.stores, self.meta = new, meta

    # ---------------------------------------------------------- readout --
    def active(self, data, pos):
        """(store, units) pairs active after reading data[pos]."""
        units = self.enc.units_at(data, pos)
        out = []
        for s in self.stores:
            if s.flags & TOP_ONLY:
                us = [u for u, t in zip(units[1:], self.enc.tables)
                      if t[0] == self.top]
                out.append((s, us))
            else:
                out.append((s, units))
        return out

    def gate(self, row, rest):
        """sigmoid(signed sqrt(cos(row, rest))), the device's gate_of."""
        dot = sum(w * rest.get(b, 0.0) for b, w in row.items())
        nr = math.sqrt(sum(w * w for w in row.values()))
        nc = math.sqrt(sum(w * w for w in rest.values()))
        c = dot / (nr * nc) if nr > 0 and nc > 0 else 0.0
        s = math.copysign(math.sqrt(abs(c)), c)
        return 1.0 / (1.0 + math.exp(-s))

    def drive(self, data, pos):
        """Positive drive per next byte. With the gated readout each
        row is scaled by its agreement with the sum of the others
        (coincidence detection); otherwise the rows are summed as the
        device's `score` kernel sums them."""
        rows = [s.row(u) for s, units in self.active(data, pos)
                for u in units]
        total = {}
        for r in rows:
            for b, w in r.items():
                total[b] = total.get(b, 0.0) + w
        if self.gated and sum(1 for r in rows if r) > 1:
            g = {}
            for r in rows:
                if not r:
                    continue
                rest = {b: total[b] - r.get(b, 0.0) for b in total}
                k = self.gate(r, rest)
                for b, w in r.items():
                    g[b] = g.get(b, 0.0) + k * w
            total = g
        return {b: v for b, v in total.items() if v > 0.0}

    def bits(self, drive, b):
        dp = sum(drive.values())
        return -math.log2((drive.get(b, 0.0) + self.lam / BYTE_UNITS)
                          / (dp + self.lam))

    def expect(self, drive, k=5):
        """The k most expected next bytes with their probabilities."""
        dp = sum(drive.values()) + self.lam
        p = [((drive.get(b, 0.0) + self.lam / BYTE_UNITS) / dp, b)
             for b in range(BYTE_UNITS)]
        p.sort(reverse=True)
        return [(b, q) for q, b in p[:k]]

    def read(self, data, learn=True):
        """Bits for every byte after the first, predicted before it is
        learned. With `learn`, each transition is then learned and the
        clock advances one tick per byte; before learning, LocusAI sleeps
        if the pressure has reached the threshold."""
        self.last_sleep = None
        if learn and self.tagged() and self.pressure() >= self.threshold:
            self.last_sleep = self.sleep(note="autonomic, pressure %.4f"
                                         % self.pressure())
        out = []
        for pos in range(len(data) - 1):
            nxt = data[pos + 1]
            out.append(self.bits(self.drive(data, pos), nxt))
            if learn:
                self.clock += 1
                for s, units in self.active(data, pos):
                    for u in units:
                        s.learn(u, nxt, self.rate, self.clock)
        return out
