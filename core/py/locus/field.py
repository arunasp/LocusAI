"""Field: continuous activation over the state space, updated as one
simultaneous event rather than a pipeline of stages.

WHY NOT A PIPELINE
An earlier design here was `excite -> spread -> kwta -> record`, and that
is wrong about the biology in a way that already produced a false result.
Cortical inhibition is not applied to a finished excitation vector; it is
part of the same dynamical system, evolving at the same time. Staging it
gave a hard top-k selection, which is a sort, which produced a step
function that was then reported as a property of the dynamics. It also
DISCARDED the sub-threshold activations, which are exactly what an
eligibility trace and a deferral confidence need to read.

So: one state, one update, every term computed from the SAME previous
state and applied together -- Jacobi, not Gauss-Seidel. No stage reads
another stage's output within a cycle. `test_field.py` asserts this by
permuting the order in which units are computed and requiring bit-
identical output, which is the only way to catch an accidental in-place
update turning the thing serial again.

THE BIOLOGICAL TARGETS, and where each is approximate rather than exact.

  Simultaneity. Real neurons are asynchronous and continuous-time; a
  discrete update is an INTEGRATION step, not a biological step. What is
  faithful is that nothing within a step depends on anything else's
  updated value.

  Two nested periods. Gamma (30-100 Hz) is described in the literature as
  a candidate unit of circuit operation, nested inside theta (3-8 Hz),
  with theta phase modulating gamma amplitude. So `step()` is the fast
  period where activation settles and `cycle()` is the slow one that
  advances. Competition belongs on the fast period, consolidation on the
  slow -- which agrees, from an independent direction, with the measured
  finding that exact trace-chain solves are consolidation-time (cubic in
  the dropped set) while truncated ones are affordable per tick.

  DEFAULT fast=7 is the theta/gamma NESTING RATIO, not a claim about how
  many items a cycle represents. That number is genuinely contested: the
  communication-through-coherence account has one item at a time with
  theta as a periodic reset, while the theta-gamma code has a sequence of
  items, one per gamma cycle. They differ precisely on concurrency, so
  this module takes no side and the ratio is a parameter.

  Shunting inhibition, not subtraction. This is the form the measurements
  forced. Purely SUBTRACTIVE inhibition (`- beta * mean`) produced a
  runaway die-off: it takes from units already at zero, the rectifier
  floors them, the mean falls, and the whole field was dead in 11-19
  steps at every beta > 0 tested. Global DIVISIVE normalisation of the
  whole state was worse in a different way -- it rescaled everything each
  step and erased the loop gain entirely, so three values of g produced
  byte-identical output. Dividing the DRIVE instead (a_j's own
  denominator grows with the rest of the field) cannot push a unit below
  rest, keeps relative differences between drives, and is the standard
  divisive-normalisation account of cortical gain control.

  The active set is a READ, not a stage. `active()` returns what is above
  threshold and leaves the field untouched; the sub-threshold values stay
  available. There is deliberately no top-k anywhere in this file.

WHAT THIS DOES NOT DO: nothing here learns. The kernel is an input and is
never modified. That is deliberate -- measuring dynamics on a fixed
kernel first means a dynamics bug cannot hide behind a learning bug,
which is a confusion that has already cost several iterations.
"""

import random

INF = float("inf")


class Field:
    """Continuous activation over ``n`` states, simultaneous update.

    Parameters map to the terms of the update rather than to stages:
      rho   self-excitation (recurrent hold). The memory term: a
            transition kernel says where activation GOES, and something
            separate has to keep it where it is against that mixing.
      g     lateral gain through the kernel.
      beta  shunting inhibition strength (divides the drive).
      leak  passive decay per unit time.
      dt    integration step. Not a biological quantity.
    """

    def __init__(self, n, kernel, rho=1.0, g=0.5, beta=0.1, leak=0.3,
                 dt=1.0, a_max=10.0, floor=1e-12, noise=0.0, seed=0,
                 areas=None):
        """Where these defaults come from, value by value.

        beta 0.1 -- MEASURED band, not a guess: the biological 1-2%%
        active band sits between 0.1 and 0.5 by participation ratio
        (2.93%% down to 0.54%%), and above 2.0 it saturates and does
        nothing further. An operating point inside a measured band, and
        the open question is what SETS the point within it.

        rho 1.0 -- the NEUTRAL value: activation is held exactly, so
        self-excitation neither grows nor decays the state on its own
        and every other term is read against it.

        g 0.5, leak 0.3, trace-like ratios -- INITIAL VALUES. The open
        question is the one the roadmap names: each should be a set
        point with feedback rather than a number, so what corrects them
        is unanswered.

        dt 1.0 -- the NEUTRAL step. Not a biological quantity at all;
        one integration step, so nothing is scaled by it.

        a_max 10.0 -- a BOUND against runaway, not a claim about the
        dynamics; `share()` exists precisely so measurements do not
        depend on it.

        floor 1e-12 -- a BOUND below which a contribution is not worth
        the multiply; it is an arithmetic cutoff, not a threshold on
        anything meaningful.

        noise 0.0 and seed 0 -- NEUTRAL defaults: spontaneous activity
        OFF unless asked for, so no run gains randomness by accident.
        """
        # THE NUMBERS ARE CHECKED, not assumed. Shape was validated
        # here and the parameters were not, so an out-of-range value
        # changed the MEANING of the dynamics instead of failing: a
        # negative beta drives `denom = 1.0 + beta * others` to zero
        # (division by zero) or below it (drive flips sign, the
        # rectifier floors every unit, and the field dies quietly --
        # the same silent collapse subtractive inhibition produced).
        # A non-finite value is worse: it propagates into every unit
        # and no later operation recovers it.
        for name, value in (("rho", rho), ("g", g), ("beta", beta),
                            ("leak", leak), ("dt", dt),
                            ("a_max", a_max), ("floor", floor),
                            ("noise", noise)):
            if value != value or value in (INF, -INF):
                raise ValueError("%s must be finite, got %r"
                                 % (name, value))
            if value < 0.0:
                raise ValueError("%s must be >= 0, got %r"
                                 % (name, value))
        if dt <= 0.0 or a_max <= 0.0:
            raise ValueError("dt and a_max must be > 0, got %r and %r"
                             % (dt, a_max))
        if len(kernel) != n:
            raise ValueError("kernel is %d x ? but n is %d"
                             % (len(kernel), n))
        for r, row in enumerate(kernel):
            if len(row) != n:
                raise ValueError("kernel row %d has %d entries, not %d"
                                 % (r, len(row), n))
        self.n = n
        self.kernel = kernel
        self.rho = rho
        self.g = g
        self.beta = beta
        self.leak = leak
        self.dt = dt
        self.a_max = a_max
        self.floor = floor
        self.a = [0.0] * n
        # Top-down bias from a SEPARATE population, not a member of this
        # pool. Biology keeps goal maintenance and pattern completion in
        # different circuits: completion is recurrent attractor dynamics
        # within a pool, while a goal is sustained elsewhere and projects
        # a biasing input into it. That distinction is why an earlier
        # measurement here was the wrong experiment -- injecting a node
        # INTO the pool and asking whether it survived found 14%, which
        # is the expected answer for a competitor and says nothing about
        # whether a goal can steer. A bias does not compete for a slot;
        # it tilts which attractor forms.
        self.bias = [0.0] * n
        # INTRINSIC EXCITABILITY, per unit, set by the unit's own activity
        # history rather than by any input (Desai, Rutherford & Turrigiano
        # 1999): a unit that wins far more than its population lowers its
        # own excitability, a silent one raises it. Enters like the bias
        # and is divided by the same inhibition; zero leaves the update
        # bit-identical. Written by a homeostatic process, never by the
        # pathway reading this field.
        self.excit = [0.0] * n
        self.steps = 0
        self.cycles = 0
        # How much of what moves this field arrives from OUTSIDE.
        # Measured rather than configured, for the same reason the bias
        # is derived rather than asserted: a constant would be an
        # assertion about the environment. An exponential average over
        # recent steps, because what matters is how much input is
        # arriving NOW -- a field being fed evidence is in a different
        # regime from one running free during replay, and the kernel is
        # only closed in the second.
        self._pending_input = 0.0
        self._input_share = 0.0
        self.input_alpha = 0.2
        # SPONTANEOUS ACTIVITY. Cortex is never silent, and this is what
        # a uniform kernel restart was a bad stand-in for: the restart
        # was deleted (inert under measurement, no biological
        # counterpart, and it removed a correct refusal), and noise is
        # the real mechanism. Together with leak it means no state traps
        # activation forever -- but it acts on the FIELD, not on the
        # kernel, so it never invents connectivity and never makes an
        # absorbing complement look admissible. A coarse level over an
        # absorbing partition still correctly does not exist.
        #
        # Seeded, and drawn BEFORE the update loop in fixed index order,
        # so the simultaneity property holds: the result must stay
        # bit-identical under a permutation of the write order.
        self.noise = noise
        self._rng = random.Random(seed)
        # PER-AREA COMPETITION. Inhibition in cortex is LOCAL: basket
        # and PV interneurons normalise within a circuit, and there is
        # no global inhibition level shared across all of cortex. A
        # single global denominator was measured to make three
        # requirements mutually exclusive -- at beta<=0.3 the whole
        # field collapses to one attractor (repertoire 1, so its 64x
        # "compression" compresses nothing into nothing), and at
        # beta>=0.7 the repertoire is ~n with compression 1.03x, so no
        # hierarchy is possible. No single beta satisfied sparsity,
        # capacity and compression together, which falsified the
        # one-control design rather than being a tuning failure.
        #
        # Areas were proposed as a second control and measured not to
        # be one: A areas divide each unit's inhibitory denominator by
        # A, scaling beta down. Active share rose from 2.3% to 16.7%
        # and the repertoire fell from 60 to 3 (see CHANGELOG.md).
        #
        # Default is a single area covering everything, which is
        # exactly the previous behaviour -- so this cannot silently
        # change an existing result.
        if areas is None:
            areas = [list(range(n))]
        seen = set()
        for a_idx, members in enumerate(areas):
            for i in members:
                if not 0 <= i < n:
                    raise ValueError("area %d has out-of-range state %d"
                                     % (a_idx, i))
                if i in seen:
                    raise ValueError("state %d is in two areas" % i)
                seen.add(i)
        if len(seen) != n:
            raise ValueError("areas cover %d of %d states"
                             % (len(seen), n))
        self.areas = [list(m) for m in areas]
        self._area_of = [0] * n
        for a_idx, members in enumerate(self.areas):
            for i in members:
                self._area_of[i] = a_idx

    # ------------------------------------------------------------ input --
    def inject(self, index, amount):  # noqa: E301
        """Add activation at one state. Injection is an EVENT, not a
        clamp: nothing re-applies it, so anything that persists
        afterwards persists on the field's own terms. Re-applying a
        constant every step is what turned an earlier measurement into a
        tautology."""
        if amount != amount or amount in (INF, -INF):
            # Evidence arrives here. A non-finite amount would spread
            # through the kernel on the next step and every figure
            # measured afterwards would be NaN with no failing step.
            raise ValueError("injected amount must be finite, got %r"
                             % (amount,))
        self.a[index] = min(self.a_max, self.a[index] + amount)
        self._pending_input += amount

    def set_bias(self, values):
        """Set the top-down bias, one value per state.

        PERSISTENT: unlike `inject`, this is not consumed. It is a
        standing input from elsewhere, so it keeps arriving every step
        until changed -- which is what maintaining a goal means, and why
        it is not the clamp that an earlier measurement mistook for
        persistence. A clamp overwrites the state; this adds an input
        that the field's own competition still has to resolve.
        """
        if len(values) != self.n:
            raise ValueError("expected %d values, got %d"
                             % (self.n, len(values)))
        self.bias = list(values)

    def set_excitability(self, values):
        """Replace the intrinsic excitability of every unit."""
        if len(values) != self.n:
            raise ValueError("excitability has %d entries, not %d"
                             % (len(values), self.n))
        self.excit = [float(v) for v in values]

    def clear_bias(self):
        self.bias = [0.0] * self.n

    def set_state(self, values):
        if len(values) != self.n:
            raise ValueError("expected %d values, got %d"
                             % (self.n, len(values)))
        self.a = [min(self.a_max, max(0.0, v)) for v in values]

    # ----------------------------------------------------------- update --
    def _drive(self, a):
        """Lateral drive through the kernel, from the GIVEN state only.

        Accumulation order is FIXED and not a parameter. It was one
        briefly, so a test could permute it -- but float addition is not
        associative, so permuting it changes the last bit of the result
        (1.0561520317937232 versus ...234 was observed) for reasons that
        have nothing to do with sequencing. A simultaneity test built on
        that would either fail spuriously or need an epsilon wide enough
        to hide a genuine sequential update. The order that MATTERS is
        the order new values are written, which `step` exposes instead.
        """
        drive = [0.0] * self.n
        for i in range(self.n):
            ai = a[i]
            if ai > self.floor:
                row = self.kernel[i]
                for j in range(self.n):
                    drive[j] += ai * row[j]
        return drive

    def step(self, order=None):
        """One fast-period update. Every term reads ``self.a`` as it was
        at entry; the new state is built separately and swapped in at the
        end, so no unit sees another unit's updated value.

        ``order`` permutes the order the NEW values are computed and
        written. Under a simultaneous update the result must be
        bit-identical whatever that order is; under a sequential sweep it
        cannot be. That is what makes it a real test rather than one
        measuring float associativity.
        """
        a = self.a
        drive = self._drive(a)
        total = sum(a)
        # Drawn up front, in fixed index order, so a permuted write
        # order cannot change which unit gets which sample.
        if self.noise > 0.0:
            spont = [self.noise * self._rng.random()
                     for _ in range(self.n)]
        else:
            spont = None
        internal = 0.0
        # Per-area sums, so the denominator below is LOCAL competition.
        area_total = [0.0] * len(self.areas)
        for i, v in enumerate(a):
            area_total[self._area_of[i]] += v
        nxt = [0.0] * self.n
        idx = range(self.n) if order is None else order
        for j in idx:
            aj = a[j]
            # Shunting against the rest of this unit's OWN AREA, so a
            # unit never inhibits itself and competition does not cross
            # area boundaries. With one area this is the whole field,
            # which is the previous behaviour exactly.
            others = area_total[self._area_of[j]] - aj
            denom = 1.0 + self.beta * others
            # The bias enters as an INPUT and is divided by the same
            # inhibition as everything else. Top-down drive competes on
            # equal terms rather than overriding -- a bias that bypassed
            # normalisation would be a clamp wearing a different name.
            net = (self.rho * aj + self.g * drive[j] + self.bias[j]
                   + self.excit[j]) / denom
            if spont is not None:
                net += spont[j]
            internal += abs(net)
            v = aj + self.dt * (net - self.leak * aj)
            if v < 0.0:
                v = 0.0
            elif v > self.a_max:
                v = self.a_max
            nxt[j] = v
        # Externally-arriving mass versus internally-generated drive,
        # for this step. Accumulated as a running share so a caller can
        # ask how grounded the field currently is.
        ext = self._pending_input
        self._pending_input = 0.0
        share = (ext / (ext + internal)) if (ext + internal) > 0 else 0.0
        self._input_share = ((1.0 - self.input_alpha) * self._input_share
                             + self.input_alpha * share)
        self.a = nxt
        self.steps += 1
        # A COPY. Returning `nxt` itself hands the caller the internal
        # state, so mutating the result corrupts the field -- caught by
        # test_step_does_not_alias_previous_state, and the same defect
        # trace_chain's full-set path already guards against.
        return list(nxt)

    def cycle(self, fast=7):
        """One slow period: ``fast`` fast updates. `fast` 7 is the
        theta/gamma NESTING RATIO taken from the literature, not a claim
        about how many items a cycle represents -- that number is
        genuinely contested (see this module's header), so it is an
        INITIAL VALUE and the open question is which account is
        right. Returns the state
        history so a caller can see whether activation settled within the
        period rather than assuming it did."""
        if fast < 1:
            raise ValueError("fast must be >= 1")
        hist = []
        for _ in range(fast):
            hist.append(list(self.step()))
        self.cycles += 1
        return hist

    # ------------------------------------------------------------ reads --
    def active(self, threshold):
        """States above ``threshold``, in index order, as (index, value).

        A READ. It does not modify the field, does not sort by value, and
        does not cap the count -- so the sub-threshold values remain
        available to anything that needs them.
        """
        return [(i, v) for i, v in enumerate(self.a) if v > threshold]

    def observed_input_share(self):
        """Share of recent drive that arrived from outside, in [0, 1).

        This is what makes an unlearned state escapable, and it is
        MEASURED, not set. A field being fed evidence reports a high
        share; one running free during replay decays toward zero, at
        which point its kernel genuinely is closed and a coarse level
        genuinely does not exist. Both are correct answers about
        different regimes, which a constant could not express.

        Capped just under 1.0 because a kernel of pure restart has no
        learned structure left in it at all.
        """
        return min(0.95, max(0.0, self._input_share))

    def total(self):
        return sum(self.a)

    def share(self, index):
        """Fraction of total activation held by one state.

        Dimensionless, so it is immune to both the saturation cap and to
        drift in the field's overall level -- which is why it is the
        right quantity to measure persistence with. A log-slope on raw
        activation is not: the trajectory here is nowhere near
        exponential, and fitting one produced a bifurcation estimate that
        had to be retracted.
        """
        t = sum(self.a)
        return (self.a[index] / t) if t > 0 else 0.0
