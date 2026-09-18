"""Cycle: the perceive-retrieve-decide-learn loop, closed.

WHAT THIS COMPOSES, and why none of it is new here. `Field` holds the
simultaneous-update dynamics (completion inside the pool, plus a bias
pathway from outside it). `Plasticity` holds eligibility traces and the
three-factor gate. `trace_chain` derives coarse levels. This file is the
wiring, and deliberately contains no new mechanism -- if a behaviour is
surprising, it belongs to one of those three and should be measured
there.

THE LOOP, one call per task step:

  perceive   present evidence as a sustained input, because the world
             keeps providing it. An injection is an event and decays;
             evidence that arrives once is not evidence about a
             situation that persists.
  settle     run the fast period until the field stops moving. THE
             DECISION IS THE SETTLED SET, not the state mid-flight --
             measured: an injected state ends up inside its own settled
             attractor only 14% of the time, so reading activation early
             reads influences rather than a decision.
  decide     read the active set. A read, never a stage, and there is no
             top-k: the sub-threshold values stay available because a
             deferral threshold and an eligibility trace both need them.
  learn      tag what co-occurred, then gate on the outcome. Activity
             alone changes nothing; only an outcome moves a weight.
  orient     derive the next bias from the learned weights, so where
             attention goes next is a consequence of outcomes rather
             than an assertion.

WHAT IT REFUSES TO DO, each because a measurement said so:

  It does not read a decision before the field settles.
  It does not set a bias directly -- `orient` derives it, so a bias
  cannot point where nothing was learned.
  It does not learn without an outcome, in either direction. A bad
  outcome is a negative modulator and depresses the same tags a good one
  would strengthen.
  It does not claim to have settled when it has not. `settled` is
  reported, not assumed, and a step that ran out of budget says so.

NOT CLAIMED: nothing here forms chunks, renders language, or decides
what an outcome is. The outcome is supplied by the caller -- for a
software task the natural source is whether the tests pass, which is
ground truth and needs no labelling.
"""

from .field import Field
from .plasticity import Plasticity
from .trace_chain import trace_chain


class LevelUp:
    """The result of forming a level. Returned rather than raised, so a
    caller can inspect a refusal as easily as a success.

    `compression` is n / chunk_count and is strictly an OUTPUT. Nothing
    in `level_up` takes a target ratio, because fixing one would mean
    choosing the number of chunks in advance and then finding states to
    fill them -- imposing a structure rather than reading the one the
    dynamics produced. The measured range is wide: 1.0x where every
    state is its own attractor, 16x where four attractors swallow 64
    states, and it moves with both inhibition and connectivity. A
    constant would be wrong at almost every operating point.
    """

    def __init__(self, admissible, reason, kernel=None,
                 representatives=None, supports=None, n_fine=0,
                 unstable=0):
        self.admissible = admissible
        self.reason = reason
        self.kernel = kernel
        self.representatives = representatives or []
        self.supports = supports or []
        self.n_fine = n_fine
        self.unstable = unstable

    @property
    def chunks(self):
        return len(self.representatives)

    @property
    def compression(self):
        return (self.n_fine / self.chunks) if self.chunks else 0.0

    def __repr__(self):
        if not self.admissible:
            return "LevelUp(refused: %s)" % self.reason
        return ("LevelUp(chunks=%d of %d, compression=%.2fx, "
                "unstable=%d)"
                % (self.chunks, self.n_fine, self.compression,
                   self.unstable))


class Step:
    """What one task step did. Returned rather than logged, so a caller
    can assert on it instead of parsing output."""

    def __init__(self, active, settled, fast_steps, tags, touched,
                 modulator):
        self.active = active
        self.settled = settled
        self.fast_steps = fast_steps
        self.tags = tags
        self.touched = touched
        self.modulator = modulator

    def __repr__(self):
        return ("Step(active=%d, settled=%s, fast=%d, tags=%d, "
                "touched=%d, mod=%s)"
                % (len(self.active), self.settled, self.fast_steps,
                   self.tags, self.touched, self.modulator))


class Cycle:
    """One pool, one plasticity store, and the loop that joins them.

    Parameters worth naming rather than defaulting silently:
      threshold   what counts as active. This is the residency dial:
                  measured, the pair store is sparse at 0.05 and
                  saturates into a dense one below 0.02, at which point
                  the whole sparse argument is void.
      beta        inhibition. The biological 1-2% active band sits
                  between 0.1 and 0.5 here (participation ratio 2.93%
                  down to 0.54%); above 2.0 it saturates and does
                  nothing further.
      bias_strength
                  peak magnitude of the derived bias. Kept well below
                  the evidence scale on purpose: a bias must steer, not
                  overrule, and at high inhibition a biased unit was
                  measured taking 92% of the field.
    """

    def __init__(self, n, kernel, threshold=0.05, beta=0.3,
                 bias_strength=0.2, rho=1.0, g=0.5, leak=0.3,
                 trace_decay=0.7, rate=0.1):
        self.n = n
        self.threshold = threshold
        self.bias_strength = bias_strength
        # ANATOMY: the connectivity the substrate starts with. Not
        # learned and not imported -- it is the structural prior, the
        # analogue of cortical local connectivity, and it is what makes
        # the field able to do anything before it has learned anything.
        self.structural = [list(row) for row in kernel]
        self.field = Field(n, self.operating_kernel(), rho=rho, g=g,
                           beta=beta, leak=leak)
        self.plasticity = Plasticity(trace_decay=trace_decay, rate=rate)
        self.steps = 0

    # -------------------------------------------------- the one kernel --
    def operating_kernel(self):
        """The kernel the field actually runs on, re-derived on demand.

        TWO COMPONENTS, and separating them is the point:
          anatomy   the structural connectivity given at construction
          learning  the weights outcomes have built, ADDED to anatomy,
                    so a negative weight (a bad outcome) suppresses a
                    structural link and a positive one strengthens it

        WHY THIS REPLACED TWO SEPARATE KERNELS, which was a real defect.
        The field used to run on the constructor kernel while `level_up`
        traced a different one, so learning reached the bias but never
        the dynamics. Now there is one kernel and learning reaches
        everything -- the dynamics, the attractors, the chunk count and
        the coarse level alike.

        A THIRD COMPONENT WAS TRIED AND DELETED: a uniform restart at a
        measured input share, meant to stop an unlearned row being
        absorbing. It was inert (`param_has_effect`: 1 distinct value of
        6, twice, across two wirings), it has no biological counterpart
        -- input arrives on specific afferents, which `inject` models --
        and it silently removed a correct refusal. Spontaneous activity
        in the field is the real mechanism; see `Field(noise=...)`.
        """
        learned = getattr(self, "plasticity", None)
        rows = []
        for i in range(self.n):
            row = list(self.structural[i])
            if learned is not None:
                for j in range(self.n):
                    w = learned.weights.get((i, j))
                    if w:
                        # Clamped at zero: a negative weight can remove
                        # a structural link but cannot invert it into
                        # negative probability.
                        row[j] = max(0.0, row[j] + w)
            s = sum(row)
            if s <= 0.0:
                row[i] = 1.0
            else:
                row = [v / s for v in row]
            rows.append(row)
        return rows

    def refresh(self):
        """Re-derive the operating kernel and hand it to the field.

        Called after learning or when the input regime has changed. Not
        called automatically on every step: the kernel is O(n^2) to
        rebuild and the dynamics are O(n^2) per step, so refreshing
        every step would double the cost for a change that is slow by
        construction -- weights move only when an outcome arrives.
        """
        self.field.kernel = self.operating_kernel()
        return self.field.kernel

    # ------------------------------------------------------------ loop --
    def settle(self, evidence=None, amount=0.2, max_fast=40,
               tol=1e-5):
        """Run the fast period until the field stops moving.

        `evidence` is a list of state indices, sustained every fast step
        rather than injected once. Returns (fast_steps, settled) where
        `settled` is False if the budget ran out -- reported rather than
        assumed, because a caller that treats an unsettled field as a
        decision is reading influences.
        """
        prev = list(self.field.a)
        for i in range(1, max_fast + 1):
            if evidence:
                for idx in evidence:
                    self.field.inject(idx, amount)
            self.field.step()
            delta = max(abs(a - b) for a, b in zip(self.field.a, prev))
            if delta < tol:
                return i, True
            prev = list(self.field.a)
        return max_fast, False

    def decide(self):
        """The active set of the settled field, as (index, value) pairs.

        A read. The values are carried out with the indices because a
        deferral threshold needs to know HOW strongly, and a hard top-k
        would have discarded exactly that.
        """
        return self.field.active(self.threshold)

    def learn(self, outcome):
        """Tag the current activation, then gate on the outcome.

        `outcome` is the third factor. Zero means no weight moves,
        whatever was tagged -- that is the property that makes this
        three-factor rather than Hebbian, and it is why a step with no
        outcome is still a legitimate step.
        """
        tags = self.plasticity.observe(self.field.a,
                                       threshold=self.threshold)
        touched = self.plasticity.consolidate(outcome)
        return tags, touched

    def orient(self, sources=None):
        """Derive the next bias from what has been learned.

        With no learned weight into a state, its bias is exactly zero,
        so attention cannot be sent somewhere nothing was built. Passing
        no `sources` uses the current active set, which makes the bias a
        function of where the system actually is.
        """
        if sources is None:
            sources = [i for i, _ in self.decide()]
        bias = self.plasticity.projected_bias(
            self.n, sources, strength=self.bias_strength)
        self.field.set_bias(bias)
        return bias

    def task_step(self, evidence, outcome=0.0, max_fast=40):
        """One full pass: perceive, settle, decide, learn, orient.

        The order is load-bearing. Learning reads the SETTLED state, so
        what gets tagged is what the field decided rather than what it
        was handed; and orient runs last, so the bias entering the next
        step is a consequence of this step's outcome.

        `refresh` runs after learning, so the kernel the next step runs
        on reflects both what was just learned and the input share just
        observed. Without it the field would keep running on the kernel
        it was constructed with, and learning would affect only the
        bias -- which is the two-kernel defect this replaced.
        """
        fast, settled = self.settle(evidence=evidence, max_fast=max_fast)
        active = self.decide()
        tags, touched = self.learn(outcome)
        if touched:
            self.refresh()
        self.orient()
        self.steps += 1
        return Step(active, settled, fast, tags, touched, outcome)

    # ------------------------------------------------------------ reads --
    def learned_kernel(self):
        """The learned weights as a kernel. Derived on every call; the
        weights remain the source of truth."""
        return self.plasticity.as_kernel(self.n)

    def reset_state(self, level=0.02):
        """Clear activation without touching what was learned. Used
        between task episodes: the point of learning is that it survives
        the episode that produced it."""
        self.field.set_state([level] * self.n)

    # ----------------------------------------------------- hierarchy --
    def survey(self, max_fast=200, hold=10, amount=0.2, stride=1):
        """Discover which stable supports the dynamics actually produce.

        Settles from each single-state injection and records the settled
        active set. A support counts only if it holds for `hold`
        consecutive fast steps -- SUPPORT stability, not value
        convergence. Those are genuinely different: measured, the
        support settles at about 45 steps while the value fixed point
        takes about 198, and a field in a limit cycle never reaches the
        latter at all. A chunk is identified by WHICH states are active
        rather than by their exact values, so support stability is the
        right criterion and the stricter one would reject usable chunks.

        Leaves the field as it found it, so surveying is a read.
        """
        found = {}
        unstable = 0
        saved = list(self.field.a)
        saved_bias = list(self.field.bias)
        for goal in range(0, self.n, stride):
            self.reset_state()
            last = None
            held = 0
            stable = None
            for _ in range(max_fast):
                self.field.inject(goal, amount)
                self.field.step()
                support = frozenset(i for i, _ in self.decide())
                if support == last:
                    held += 1
                    if held >= hold:
                        stable = support
                        break
                else:
                    held = 0
                    last = support
            if stable is None:
                unstable += 1
            elif stable:
                found.setdefault(stable, []).append(goal)
        self.field.set_state(saved)
        self.field.set_bias(saved_bias)
        return found, unstable

    def level_up(self, max_fast=200, hold=10, stride=1):
        """Form the next level from the chunks the dynamics produced.

        A CHUNK IS A STABLE ATTRACTOR SUPPORT, and the coarse kernel is
        the trace chain over one representative per chunk. That is not a
        new mechanism: `trace_chain` is already the level-coupling
        operator, and its transitivity -- tested to 9 decimal places --
        is what makes applying it to its own output well defined.

        REPRESENTATIVES RATHER THAN LUMPING THE SUPPORTS, because naive
        block-averaging was measured against the exact trace and does
        NOT equal it; worse, lumping is the one that gets the long-run
        measure wrong. Tracing to a representative set is exact.

        `input_rate` is gone. A uniform restart was mixed into the
        kernel so an unlearned row would not be absorbing, and it is
        deleted: inert under measurement, absent from biology, and it
        silently removed this refusal. AN ABSORBING COMPLEMENT MEANS NO
        COARSE LEVEL EXISTS over that partition, which is information
        rather than an error. What keeps a real substrate from trapping
        activation is leak plus spontaneous activity, so the answer is
        `Field(noise=...)` and not a kernel patch.

        TWO PRECONDITIONS, both refused rather than worked around.
        Nothing learned means nothing to chunk, so a store with no
        weights is refused explicitly rather than allowed to produce
        chunks out of a structureless kernel. And the partition must be
        ADMISSIBLE -- `trace_chain` refuses an inadmissible subset
        rather than returning a non-stochastic matrix, so that
        condition is enforced by the operator instead of by a number
        someone chose.
        """
        if self.plasticity.live_weights() == 0:
            return LevelUp(False, "nothing learned: a level formed from "
                                  "a structureless kernel would be "
                                  "chunks of nothing", n_fine=self.n)
        supports, unstable = self.survey(max_fast=max_fast, hold=hold,
                                         stride=stride)
        if not supports:
            return LevelUp(False, "no stable support from any injection",
                           n_fine=self.n, unstable=unstable)
        # One representative per chunk: the member holding the most
        # activation, skipping any already claimed, since supports can
        # overlap and a representative must name exactly one chunk.
        taken = set()
        reps = []
        kept = []
        for support in sorted(supports, key=lambda s: (-len(s),
                                                       sorted(s))):
            members = sorted(support, key=lambda i: -self.field.a[i])
            pick = next((m for m in members if m not in taken), None)
            if pick is None:
                continue
            taken.add(pick)
            reps.append(pick)
            kept.append(support)
        if len(reps) < 2:
            return LevelUp(False, "only %d distinct chunk(s) -- a level "
                                  "of one state has no dynamics"
                                  % len(reps), n_fine=self.n,
                           representatives=reps, supports=kept,
                           unstable=unstable)
        fine = self.operating_kernel()
        try:
            coarse = trace_chain(fine, sorted(reps))
        except ValueError as exc:
            return LevelUp(False, "inadmissible partition: %s" % exc,
                           n_fine=self.n, representatives=reps,
                           supports=kept, unstable=unstable)
        return LevelUp(True, "ok", kernel=coarse,
                       representatives=sorted(reps), supports=kept,
                       n_fine=self.n, unstable=unstable)

    def next_level(self, result, **kwargs):
        """A Cycle over the coarse kernel, so levels compose.

        Deliberately a NEW Cycle with its OWN plasticity store rather
        than a view: a coarse level learns its own weights from its own
        activity, and sharing a store would let fine-level tags
        consolidate into coarse-level weights -- the import path wearing
        a hierarchy.
        """
        if not result.admissible:
            raise ValueError("cannot build a level from a refusal: %s"
                             % result.reason)
        params = {
            "threshold": self.threshold,
            "bias_strength": self.bias_strength,
        }
        params.update(kwargs)
        return Cycle(len(result.representatives), result.kernel,
                     **params)
