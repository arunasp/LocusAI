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
                 dt=1.0, a_max=10.0, floor=1e-12):
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
        self.steps = 0
        self.cycles = 0

    # ------------------------------------------------------------ input --
    def inject(self, index, amount):
        """Add activation at one state. Injection is an EVENT, not a
        clamp: nothing re-applies it, so anything that persists
        afterwards persists on the field's own terms. Re-applying a
        constant every step is what turned an earlier measurement into a
        tautology."""
        self.a[index] = min(self.a_max, self.a[index] + amount)

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
        nxt = [0.0] * self.n
        idx = range(self.n) if order is None else order
        for j in idx:
            aj = a[j]
            # Shunting: the denominator is the REST of the field, so a
            # unit never inhibits itself and can never be driven below
            # rest by inhibition.
            others = total - aj
            denom = 1.0 + self.beta * others
            net = (self.rho * aj + self.g * drive[j]) / denom
            v = aj + self.dt * (net - self.leak * aj)
            if v < 0.0:
                v = 0.0
            elif v > self.a_max:
                v = self.a_max
            nxt[j] = v
        self.a = nxt
        self.steps += 1
        # A COPY. Returning `nxt` itself hands the caller the internal
        # state, so mutating the result corrupts the field -- caught by
        # test_step_does_not_alias_previous_state, and the same defect
        # trace_chain's full-set path already guards against.
        return list(nxt)

    def cycle(self, fast=7):
        """One slow period: ``fast`` fast updates. Returns the state
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
