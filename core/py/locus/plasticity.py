"""Plasticity: eligibility traces plus a three-factor rule.

THE DEFINING PROPERTY, and the one thing that makes this three-factor
rather than Hebbian: ACTIVITY ALONE CHANGES NO WEIGHT. Coincident pre-
and post-synaptic activity sets a local, decaying TAG. A weight moves
only when a broadcast modulator arrives while that tag is still alive.
Plain Hebbian learning writes the weight immediately, which makes every
coincidence permanent and gives no way to say which coincidences
mattered. `test_activity_alone_changes_nothing` is the test that would
catch a drift back to that.

THE BIOLOGICAL TARGETS, and where each is approximate.

  Eligibility trace. A synapse-local tag from coincident activity, with
  its own decay. Faithful in shape: the update reads only that synapse's
  own pre and post values, which `test_trace_is_local` asserts by
  perturbing an unrelated unit and requiring the trace to be unchanged.
  Approximate in that a real tag is a molecular state with saturation
  and a stochastic lifetime; this is a scalar with exponential decay.

  Synaptic tagging and capture (Frey & Morris). A weakly stimulated
  synapse sets a tag; plasticity-related product made elsewhere is
  CAPTURED by whichever synapses are currently tagged. That is exactly a
  local tag plus a broadcast third factor, and it is why the time window
  matters: a modulator arriving after the tag has decayed finds nothing
  to capture. `test_modulator_after_decay_does_little` holds that.

  The third factor is a SCALAR. One value for the whole field, not a
  per-synapse quantity. On the hardware measured for this project that
  is also the cheap shape -- a wave-uniform value lives in a scalar
  register and costs no vector bandwidth -- so the biology and the
  device agree here rather than trading off.

  Sign. A modulator can be negative, and the same tag then depresses
  instead of potentiating. One signal, two directions, no separate
  depression rule.

  Homeostatic scaling, on the SLOW period only. Real synaptic scaling is
  slow and multiplicative: it changes all of a neuron's weights by a
  common factor, preserving their relative sizes. That distinction is
  not pedantic here -- a global rescale applied per step is precisely
  what erased the loop gain in an earlier measurement, where three
  values of the gain parameter produced byte-identical output. So
  scaling is multiplicative, per-target, and called from the slow period
  rather than every update.

STREAMS NOT IMPORTS. Weights start EMPTY and only ever come from
observed activity. There is no constructor argument that installs them,
no path that seeds from a kernel, embedding or co-occurrence count.
`test_nothing_is_installed` exists to keep that true, because it is the
one property that cannot be recovered once lost.

SPARSE BY CONSTRUCTION. Traces and weights live in dicts keyed by
(pre, post). A dense n-by-n pair store is the thing that does not scale:
at the sizes this project is aiming at, an explicit edge list for every
possible pair is the dominant cost and almost all of it is zero.
"""


class Plasticity:
    """Eligibility traces and gated weight updates over pair keys.

    Parameters:
      trace_decay  per-step multiplier on every live tag. Sets the window
                   in which a modulator can still find something.
      rate         learning rate applied to (tag * modulator).
      trace_floor  tags below this are dropped rather than kept as
                   near-zero entries, which is what stops the dict
                   growing without bound.
      w_max        magnitude cap per weight.
    """

    def __init__(self, trace_decay=0.7, rate=0.1, trace_floor=1e-4,
                 w_max=5.0):
        if not 0.0 < trace_decay < 1.0:
            raise ValueError("trace_decay must be in (0, 1)")
        self.trace_decay = trace_decay
        self.rate = rate
        self.trace_floor = trace_floor
        self.w_max = w_max
        # Both EMPTY. Nothing is installed; see the module note.
        self.traces = {}
        self.weights = {}
        # Outgoing targets per source, so a per-source pass visits only
        # its own rows. Derived from `weights`, never set directly.
        self._out = {}
        self.observations = 0
        self.consolidations = 0

    # ------------------------------------------------------------ fast --
    def observe(self, activation, threshold=0.0):
        """Decay every tag, then add coincident pre*post for active pairs.

        Called on the FAST period. Reads activation and nothing else, and
        deliberately does NOT touch weights -- that separation is the
        three-factor property, not an implementation convenience.

        Only pairs whose pre AND post both exceed ``threshold``
        contribute, which is what keeps this O(active^2) rather than
        O(n^2). The active set is small by design, so that is the whole
        reason the sparse form is affordable.
        """
        decayed = {}
        for key, tag in self.traces.items():
            t = tag * self.trace_decay
            if abs(t) > self.trace_floor:
                decayed[key] = t
        live = [(i, v) for i, v in enumerate(activation) if v > threshold]
        for i, pre in live:
            for j, post in live:
                if i == j:
                    continue
                key = (i, j)
                decayed[key] = decayed.get(key, 0.0) + pre * post
        self.traces = decayed
        self.observations += 1
        return len(self.traces)

    def observe_transition(self, before, after):
        """Directional tags: (i -> j) for i active BEFORE and j AFTER.

        Decays every tag, then adds pre * post for each ordered pair,
        pre from ``before`` and post from ``after`` (both lists of
        (unit, value)). Only the forward direction is tagged; this is the
        pre-before-post ordering of spike-timing-dependent plasticity,
        which `observe` cannot express because it pairs units active at
        the same moment in both directions. A unit may follow itself.

        Like `observe`, it never touches a weight.
        """
        decayed = {}
        for key, tag in self.traces.items():
            t = tag * self.trace_decay
            if abs(t) > self.trace_floor:
                decayed[key] = t
        for i, pre in before:
            for j, post in after:
                key = (i, j)
                decayed[key] = decayed.get(key, 0.0) + pre * post
        self.traces = decayed
        self.observations += 1
        return len(self.traces)

    # ------------------------------------------------------------ slow --
    def consolidate(self, modulator):
        """Apply ``rate * tag * modulator`` to every tagged weight.

        The gate. With ``modulator`` zero nothing changes, whatever the
        tags hold -- which is the property that distinguishes this from
        Hebbian learning. A negative modulator depresses using the same
        tags rather than needing a separate rule.

        Returns the number of weights touched, so a caller can tell a
        no-op apart from a silent failure.
        """
        if modulator == 0.0:
            return 0
        touched = 0
        for key, tag in self.traces.items():
            dw = self.rate * tag * modulator
            if dw == 0.0:
                continue
            if key not in self.weights:
                self._out.setdefault(key[0], set()).add(key[1])
            w = self.weights.get(key, 0.0) + dw
            if w > self.w_max:
                w = self.w_max
            elif w < -self.w_max:
                w = -self.w_max
            self.weights[key] = w
            touched += 1
        self.consolidations += 1
        return touched

    def scale_to(self, target_sum):
        """Homeostatic scaling: multiplicative, per POST-synaptic target.

        Slow-period only. Multiplicative rather than subtractive so the
        relative sizes of a target's incoming weights are preserved --
        the property a global rescale destroys. Returns how many targets
        were rescaled.
        """
        if target_sum <= 0.0:
            raise ValueError("target_sum must be positive")
        incoming = {}
        for (i, j), w in self.weights.items():
            incoming.setdefault(j, []).append((i, w))
        rescaled = 0
        for j, pairs in incoming.items():
            total = sum(abs(w) for _, w in pairs)
            if total <= 0.0:
                continue
            factor = target_sum / total
            if factor == 1.0:
                continue
            for i, w in pairs:
                self.weights[(i, j)] = w * factor
            rescaled += 1
        return rescaled

    def scale_sources(self, target_sum, sources=None):
        """Homeostatic scaling per PRE-synaptic source: multiplicative, so
        a source's outgoing weights keep their relative sizes while their
        absolute sum becomes ``target_sum``. Candidate targets of one
        source then compete for a fixed total, which is how a learned
        association is favoured without a negative modulator ever
        depressing it.

        Slow-period only, like `scale_to`. ``sources`` restricts the pass
        to those rows; None scales every source. Returns how many rows
        were rescaled.
        """
        if target_sum <= 0.0:
            raise ValueError("target_sum must be positive")
        rows = self._out if sources is None else {
            i: self._out[i] for i in set(sources) if i in self._out}
        outgoing = {i: [(j, self.weights[(i, j)]) for j in targets]
                    for i, targets in rows.items()}
        rescaled = 0
        for i, pairs in outgoing.items():
            total = sum(abs(w) for _, w in pairs)
            if total <= 0.0:
                continue
            factor = target_sum / total
            if factor == 1.0:
                continue
            for j, w in pairs:
                self.weights[(i, j)] = w * factor
            rescaled += 1
        return rescaled

    # ----------------------------------------------------------- reads --
    def weight(self, pre, post):
        return self.weights.get((pre, post), 0.0)

    def trace(self, pre, post):
        return self.traces.get((pre, post), 0.0)

    def live_traces(self):
        return len(self.traces)

    def live_weights(self):
        return len(self.weights)

    def projected_bias(self, n, sources, strength=1.0):
        """Bias DERIVED from learned weights, never set by a caller.

        bias[j] = strength * (sum of w[i][j] for i in sources), scaled so
        the largest magnitude is `strength`.

        WHY THIS EXISTS RATHER THAN A SETTABLE BIAS. A hand-set bias is
        an assertion, and a measurement here showed what that costs: a
        bias at double strength on a state with NO structural support
        beat sustained evidence on a supported pattern, 0.588 against
        0.406. Expectation manufacturing structure the substrate never
        formed is the dynamical form of the import path this project
        exists to close.

        Deriving it fixes that by construction rather than by tuning. A
        state with no learned incoming weight from the sources gets
        EXACTLY zero, so a bias cannot point somewhere nothing was
        learned. Where it points, and how strongly, is then a
        consequence of outcomes -- because that is the only thing that
        moves a weight (see `consolidate`).

        SUPPRESSION NEEDS NO SEPARATE RULE. A bad outcome is a negative
        modulator, which depresses the tagged pairs, which yields a
        NEGATIVE bias here. The same mechanism that steers toward a good
        outcome steers away from a bad one.

        Scaling by the peak rather than the sum keeps `strength` the
        maximum magnitude regardless of how many weights contribute, so
        a well-learned region cannot produce an arbitrarily large bias
        just by having more edges.
        """
        bias = [0.0] * n
        for i in sources:
            for j in range(n):
                w = self.weights.get((i, j))
                if w:
                    bias[j] += w
        peak = max((abs(v) for v in bias), default=0.0)
        if peak > 0.0:
            bias = [strength * v / peak for v in bias]
        return bias

    def as_kernel(self, n, floor=0.0):
        """Learned weights as a dense row-normalised kernel, for feeding
        back into a Field or a trace-chain computation.

        DERIVED, never stored: this builds a new matrix on each call and
        the weights remain the source of truth. Rows with no outgoing
        weight get a self-loop rather than a uniform row, because a
        uniform row would invent connectivity the substrate never
        formed.

        A `input_rate` parameter briefly lived here, mixing in a uniform
        restart so an unlearned row would not be absorbing. IT IS
        DELETED, on three counts. Biologically there is no uniform
        restart: nothing makes activation jump to a random state, and
        external input arrives on SPECIFIC afferents -- which `inject`
        already models. Empirically it was inert, failing
        `param_has_effect` at 1 distinct value of 6 across two
        different wirings. And it silently removed a correct refusal:
        with every row carrying mass, nothing was ever absorbing, so a
        coarse level formed where none should exist.

        An absorbing complement means NO COARSE LEVEL EXISTS over that
        partition. That is information, not an error to be patched. What
        keeps a real substrate from trapping activation is leak plus
        SPONTANEOUS ACTIVITY -- see `Field.noise`, which is the
        biological mechanism the restart was standing in for.
        """
        rows = []
        for i in range(n):
            row = [0.0] * n
            for j in range(n):
                w = self.weights.get((i, j), 0.0)
                if w > floor:
                    row[j] = w
            s = sum(row)
            if s <= 0.0:
                row[i] = 1.0
            else:
                row = [v / s for v in row]
            rows.append(row)
        return rows
