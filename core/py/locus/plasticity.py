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

INF = float("inf")


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
        """trace_decay 0.7 and rate 0.1 are INITIAL VALUES: the decay
        sets how long a tag stays eligible and the rate how far one
        outcome moves a weight, and neither is measured here. The open
        question for both is the roadmap's -- a set point with feedback
        rather than a constant. trace_floor 1e-4 is a BOUND below which
        a tag is dropped rather than carried, arithmetic not biology.
        w_max 5.0 is a BOUND against runaway, paired with the clip in
        knowledge.py.
        """
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

        `threshold` 0.0 is the NEUTRAL default: every non-zero
        activation counts, so the caller decides what counts as active
        rather than inheriting a cutoff from here.

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

    def observe_error(self, before, errors):
        """Signed tags from a local prediction error.

        (i -> j) gets pre_i * err_j, where ``errors`` holds (unit, actual
        minus predicted) for post units. The prediction is the post
        unit's own, formed from its inputs, so the term stays local to
        the synapse's two ends: the dendritic prediction of somatic
        activity (Urbanczik & Senn 2014) and the Rescorla-Wagner error.
        A post unit that was predicted but stayed silent gets a negative
        tag, so depression needs no separate rule. Never touches a
        weight.
        """
        decayed = {}
        for key, tag in self.traces.items():
            t = tag * self.trace_decay
            if abs(t) > self.trace_floor:
                decayed[key] = t
        for i, pre in before:
            for j, err in errors:
                if err == 0.0:
                    continue
                key = (i, j)
                decayed[key] = decayed.get(key, 0.0) + pre * err
        self.traces = decayed
        self.observations += 1
        return len(self.traces)

    # ------------------------------------------------------------ slow --
    def consolidate(self, modulator, consume=False, bound=None,
                    local=None):
        """Apply ``rate * tag * modulator`` to every tagged weight.

        The gate. With ``modulator`` zero nothing changes, whatever the
        tags hold -- which is the property that distinguishes this from
        Hebbian learning. A negative modulator depresses using the same
        tags rather than needing a separate rule.

        Returns the number of weights touched, so a caller can tell a
        no-op apart from a silent failure.

        ``consume``: the tags are captured by this modulator and cleared,
        so a later modulator cannot apply them again (synaptic tagging
        and capture). ``bound``: soft bounds on [0, bound] -- potentiation
        scales by (1 - w / bound) and depression by (w / bound), so a
        weight saturates instead of being clipped. This is the soft-bound
        form A+(w) = (w_max - w) eta+, A-(w) = w eta- in Gerstner's STDP
        review; van Rossum, Bi & Turrigiano (2000) use weight-dependent
        depression with additive potentiation. ``local`` maps a source
        unit to a multiplier on its synapses' changes: a synapse-local
        factor such as metaplasticity, where a synapse's own history sets
        how plastic it is; unlisted sources get 1. Without these
        arguments, behaviour is unchanged.
        """
        # THE GATE IS THIS BRANCH, not the arithmetic below it. With a
        # zero modulator `dw = rate * tag * 0.0` also comes out zero, so
        # the two agree today and removing this line changes nothing --
        # which is exactly why it is worth stating: the property must
        # not rest on a multiplication. Any term added inside the loop
        # that does not itself carry the modulator (a decay, a floor, a
        # bias) would move weights with NO OUTCOME, and only this branch
        # stops it. `test_a_zero_modulator_never_looks_at_the_tags`
        # binds it, so the line cannot be deleted as redundant.
        if modulator != modulator or modulator in (INF, -INF):
            # A non-finite modulator is NOT caught by the zero test and
            # would write NaN into every tagged weight, silently and
            # irreversibly. Refused rather than propagated.
            raise ValueError("modulator must be finite, got %r"
                             % (modulator,))
        if modulator == 0.0:
            return 0
        touched = 0
        for key, tag in self.traces.items():
            dw = self.rate * tag * modulator
            if local is not None:
                dw *= local.get(key[0], 1.0)
            if dw == 0.0:
                continue
            if bound is not None:
                old = self.weights.get(key, 0.0)
                dw *= (1.0 - old / bound) if dw > 0.0 else old / bound
                if dw == 0.0:
                    continue
            if key not in self.weights:
                self._out.setdefault(key[0], set()).add(key[1])
            w = self.weights.get(key, 0.0) + dw
            if bound is not None:
                w = min(max(w, 0.0), bound)
            if w > self.w_max:
                w = self.w_max
            elif w < -self.w_max:
                w = -self.w_max
            self.weights[key] = w
            touched += 1
        if consume:
            self.traces = {}
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

    def scale_targets(self, factors):
        """Homeostatic scaling per POST-synaptic target by a given factor.

        ``factors`` maps a target to the multiplier for all its incoming
        weights, so their relative sizes are kept (Turrigiano et al. 1998). The
        caller derives each factor from that target's own activity
        against its set point; nothing here chooses one. Returns how many
        weights changed.
        """
        changed = 0
        for (i, j), w in self.weights.items():
            f = factors.get(j)
            if f is None or f == 1.0 or f <= 0.0:
                continue
            self.weights[(i, j)] = w * f
            changed += 1
        return changed

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
