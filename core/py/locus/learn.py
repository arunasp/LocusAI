"""Learners: plasticity rules driven by an encoded byte stream.

Each learner owns a `Plasticity` and an encoder (`locus.encode`) and
learns next-byte structure one position at a time: the context units
active after byte t predict byte t + 1. `score` reads any learner out in
bits per byte, so learners are compared on held-out text rather than on
their own training signal.

Readout: drive[b] is the sum of learned weights from the context units
to byte unit b; p(b) = (max(drive[b], 0) + lam / 256) / (positive drive
+ lam). `lam` is the only readout parameter; callers fit it on held-out
validation text, never on the text they report.

Nothing here installs a weight: every weight comes from `consolidate`
(CONSTITUTION.md, streams not imports).
"""

import math
import random

from .encode import BYTE_UNITS

INF = float("inf")


def drive_of(p, units):
    """Summed learned weight from ``units`` to each byte unit."""
    drive = {}
    for u in units:
        for b in p._out.get(u, ()):
            if b < BYTE_UNITS:
                drive[b] = drive.get(b, 0.0) + p.weights[(u, b)]
    return drive


def prob(drive, b, lam):
    """p(b) from a drive, with ``lam`` of uniform mass mixed in."""
    pos = sum(v for v in drive.values() if v > 0.0)
    return (max(drive.get(b, 0.0), 0.0) + lam / BYTE_UNITS) / (pos + lam)


def score_with(drive_fn, enc, files, lam):
    """(bits per byte, predictions) of ``drive_fn(units)`` on ``files``.

    Checked HERE rather than inside `prob`, which runs once per byte.
    A non-positive lam makes an unseen byte impossible (-log2(0)), and
    a negative one leaves the distribution summing to 1 while single
    probabilities go negative -- so the bits figure either raises or
    comes out better than physically possible.
    """
    if lam != lam or lam in (INF, -INF) or lam <= 0.0:
        raise ValueError("lam must be a finite positive number, got %r"
                         % (lam,))
    bits = 0.0
    n = 0
    for data in files:
        for pos in range(len(data) - 1):
            b = data[pos + 1]
            drive = drive_fn(enc.units_at(data, pos))
            bits -= math.log2(prob(drive, b, lam))
            n += 1
    return bits / max(n, 1), n


def score(p, enc, files, lam):
    """(bits per byte, predictions) of plasticity ``p`` on ``files``."""
    return score_with(lambda units: drive_of(p, units), enc, files, lam)


class TagLearner:
    """Ordered tags with a scalar modulator.

    Per position: tag context -> next byte (`observe_transition`), then
    `consolidate` with the surprise of the next byte (``modulated``) or
    1. Each source row is rescaled to sum 1 at every file end
    (``per_file``), or once after all files.
    """

    def __init__(self, plasticity, encoder, modulated=True, per_file=True):
        self.p = plasticity
        self.enc = encoder
        self.modulated = modulated
        self.per_file = per_file

    def learn(self, files):
        """Learn from ``files`` in order; returns online bits per byte."""
        p, enc = self.p, self.enc
        bits = 0.0
        n = 0
        for data in files:
            touched = set()
            for pos in range(len(data) - 1):
                units = enc.units_at(data, pos)
                nxt = data[pos + 1]
                s = -math.log2(prob(drive_of(p, units), nxt, 1.0))
                bits += s
                n += 1
                p.observe_transition([(u, 1.0) for u in units],
                                     [(nxt, 1.0)])
                p.consolidate(s if self.modulated else 1.0)
                touched.update(units)
            if self.per_file:
                p.scale_sources(1.0, sources=touched)
        if not self.per_file:
            p.scale_sources(1.0)
        return bits / max(n, 1)


class FaithfulLearner:
    """Delta rule with a local error, a noradrenergic gain, tag capture,
    soft bounds on [0, 1] and per-target homeostasis at each file end.

    Error for post byte b: (1 if b is the next byte else 0) - p(b), over
    the bytes the context predicts plus the actual one; all context units
    share it. Gain: surprise over the mean of all earlier surprise (1
    when off or first). Homeostasis: each byte unit's incoming weights are
    scaled by its actual count over its summed predicted probability.
    """

    def __init__(self, plasticity, encoder, ne_gain=True, homeostasis=True):
        self.p = plasticity
        self.enc = encoder
        self.ne_gain = ne_gain
        self.homeostasis = homeostasis

    def learn(self, files):
        """Learn from ``files`` in order; returns online bits per byte."""
        p, enc = self.p, self.enc
        bits = 0.0
        n = 0
        mean_s = 0.0
        for data in files:
            predicted = {}
            actual = {}
            for pos in range(len(data) - 1):
                units = enc.units_at(data, pos)
                nxt = data[pos + 1]
                drive = drive_of(p, units)
                targets = set(drive) | {nxt}
                probs = {b: prob(drive, b, 1.0) for b in targets}
                s = -math.log2(probs[nxt])
                bits += s
                n += 1
                gain = s / mean_s if (self.ne_gain and mean_s > 0.0) else 1.0
                mean_s += (s - mean_s) / n
                errors = [(b, (1.0 if b == nxt else 0.0) - q)
                          for b, q in probs.items()]
                p.observe_error([(u, 1.0) for u in units], errors)
                p.consolidate(gain, consume=True, bound=1.0)
                for b in drive:
                    predicted[b] = predicted.get(b, 0.0) + probs[b]
                actual[nxt] = actual.get(nxt, 0) + 1
            if self.homeostasis:
                p.scale_targets({b: actual[b] / q
                                 for b, q in predicted.items()
                                 if q > 0.0 and actual.get(b)})
        return bits / max(n, 1)


def branch_update(p, units, nxt, gain, rates=None):
    """One per-input delta step: each unit in ``units`` moves its own row
    toward the next byte by its own error, 1[b is nxt] - w[u -> b], over
    its row plus ``nxt``. ``rates`` maps a unit to a multiplier on its
    change (see `consolidate(local=)`); None leaves every unit at 1.
    """
    for u in units:
        row = {b for b in p._out.get(u, ()) if b < BYTE_UNITS}
        row.add(nxt)
        errors = [(b, (1.0 if b == nxt else 0.0)
                   - p.weights.get((u, b), 0.0)) for b in row]
        p.observe_error([(u, 1.0)], errors)
        local = None if rates is None else {u: rates[u]}
        p.consolidate(gain, consume=True, local=local)


def unit_order(enc, u):
    """n-gram order of unit ``u`` in encoder ``enc``; 1 for byte units."""
    if u < BYTE_UNITS:
        return 1
    for k, _h, _seed, offset, size in enc.tables:
        if offset <= u < offset + size:
            return k
    raise ValueError("unit %d is outside every table" % u)


class BranchLearner:
    """Delta rule with a separate error per input unit.

    Each context unit u predicts the next byte from its own row alone,
    p_u(b) = w[u -> b], and learns from its own error 1[b is next] -
    w[u -> b] over its row plus the actual byte. Inputs therefore do not
    share one error or compete to explain a byte, as in
    `FaithfulLearner`; each behaves like a dendritic branch with its own
    prediction.

    ``metaplastic``: the change on a unit's synapses is scaled so its
    effective rate is 1 / (times the unit has been active), through
    `consolidate(local=...)`. The first observation is learned in one
    step and later ones adjust less, and each row is exactly the running
    frequency of the bytes that followed the unit. Otherwise the rate is
    the plasticity's constant `rate`. ``ne_gain`` as in
    `FaithfulLearner`, from the summed readout.
    """

    def __init__(self, plasticity, encoder, metaplastic=False,
                 ne_gain=False):
        self.p = plasticity
        self.enc = encoder
        self.metaplastic = metaplastic
        self.ne_gain = ne_gain
        self.seen = {}

    def learn(self, files):
        """Learn from ``files`` in order; returns online bits per byte."""
        p, enc = self.p, self.enc
        bits = 0.0
        n = 0
        mean_s = 0.0
        for data in files:
            for pos in range(len(data) - 1):
                units = enc.units_at(data, pos)
                nxt = data[pos + 1]
                s = -math.log2(prob(drive_of(p, units), nxt, 1.0))
                bits += s
                n += 1
                gain = s / mean_s if (self.ne_gain and mean_s > 0.0) else 1.0
                mean_s += (s - mean_s) / n
                rates = None
                if self.metaplastic:
                    rates = {}
                    for u in units:
                        k = self.seen.get(u, 0) + 1
                        self.seen[u] = k
                        rates[u] = 1.0 / (p.rate * k)
                branch_update(p, units, nxt, gain, rates)
        return bits / max(n, 1)


class CLSLearner:
    """Complementary learning systems (McClelland, McNaughton & O'Reilly
    1995): a fast hippocampal store beside a slow neocortex, linked by
    replay.

    Hippocampus (``hippo``): only the highest-order n-gram units, the
    most specific contexts, with per-input errors at the metaplastic rate
    1/n -- learned in one step on first sight, the running frequency
    after that.

    Neocortex (``cortex``): every context unit, per-input errors at the
    plasticity's constant `rate`.

    Replay (``replay``): at each file end, the offline period, the cortex
    relearns that file's episodes interleaved with as many earlier
    episodes, in a seeded shuffle. Episodes are kept as (file, position)
    and re-encoded when replayed.

    Readout: `drive` sums the hippocampal and cortical drives; nothing
    weights one system against the other. With ``hippocampus`` off and
    no replay the learner is `BranchLearner` at a constant rate.
    """

    def __init__(self, hippo, cortex, encoder, replay=True, seed=1,
                 hippocampus=True):
        self.hippo = hippo
        self.cortex = cortex
        self.enc = encoder
        self.replay = replay
        self.hippocampus = hippocampus
        self.seed = seed
        self.top = max(encoder.orders)
        self.seen = {}

    def hippo_units(self, units):
        if not self.hippocampus:
            return []
        return [u for u in units if unit_order(self.enc, u) == self.top]

    def drive(self, units):
        """Summed hippocampal and cortical drive for ``units``."""
        d = drive_of(self.cortex, units)
        for b, v in drive_of(self.hippo, self.hippo_units(units)).items():
            d[b] = d.get(b, 0.0) + v
        return d

    def learn(self, files):
        """Learn from ``files`` in order; returns online bits per byte."""
        rnd = random.Random(self.seed)
        enc, h = self.enc, self.hippo
        bits = 0.0
        n = 0
        earlier = []
        for fi, data in enumerate(files):
            episodes = []
            for pos in range(len(data) - 1):
                units = enc.units_at(data, pos)
                nxt = data[pos + 1]
                bits -= math.log2(prob(self.drive(units), nxt, 1.0))
                n += 1
                hu = self.hippo_units(units)
                rates = {}
                for u in hu:
                    k = self.seen.get(u, 0) + 1
                    self.seen[u] = k
                    rates[u] = 1.0 / (h.rate * k)
                branch_update(h, hu, nxt, 1.0, rates)
                branch_update(self.cortex, units, nxt, 1.0)
                episodes.append((fi, pos))
            if self.replay and episodes:
                old = (rnd.sample(earlier, min(len(earlier), len(episodes)))
                       if earlier else [])
                batch = episodes + old
                rnd.shuffle(batch)
                for ei, pos in batch:
                    d = files[ei]
                    branch_update(self.cortex, enc.units_at(d, pos),
                                  d[pos + 1], 1.0)
            earlier.extend(episodes)
        return bits / max(n, 1)
