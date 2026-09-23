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
      beta        inhibition. Measured in one configuration per beta
                  (`make operating-point`, n=64), the active share
                  reaches the biological 1-2% band at beta >= 0.7;
                  beta 0.5 leaves 10.9% active.
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
    def reset_state(self, level=0.02):
        """Clear activation without touching what was learned. Used
        between task episodes: the point of learning is that it survives
        the episode that produced it."""
        self.field.set_state([level] * self.n)

    # ----------------------------------------------------- hierarchy --
    def survey(self, max_fast=200, hold=10, amount=0.2, stride=1):
        """Discover which stable supports the dynamics actually produce.

        `stride` DEFAULTS TO 1, and that default is load-bearing rather
        than tidy. You cannot discover more attractors than you have
        injection points, so a stride of k caps the support count at
        n/k BY CONSTRUCTION. With stride=8 the chunk counts came back 8
        of 8 samples, 15 of 16, 27 of 32, 43 of 64 -- tracking the
        sampling almost exactly -- and the resulting compression was
        reported as 8.00x, 8.53x and 10.67x and read as an emergent
        property of the dynamics. At stride=1 the real figures are
        1.49x at n=64 and 1.24x at n=128. That is a five- to sevenfold
        overstatement AND a different regime: most states turn out to
        be their own attractor, which is capacity-maximum and
        compression-near-zero.

        A larger stride is still accepted for a quick look, but any
        number taken from it must be checked with
        `watchdog.not_harness_bound` against n/stride before it is
        believed. `param_has_effect` is not sufficient here and did in
        fact pass: the values differed across conditions, they just
        differed because the harness parameter differed.
        """
        found = {}
        unstable = 0
        saved = list(self.field.a)
        saved_bias = list(self.field.bias)
        # Support stability, NOT value convergence. They differ: the
        # support settles at about 45 steps while the value fixed point
        # takes about 198, and a field in a limit cycle never reaches
        # the latter at all. A chunk is identified by WHICH states are
        # active rather than by their values, so the stricter criterion
        # would reject usable chunks. Leaves the field as it found it,
        # so surveying is a read.
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

    def survey_driven(self, stream, max_fast=200, hold=10, amount=0.2):
        """The repertoire EXPERIENCE carved, not one an experimenter
        enumerated.

        WHY THE ENUMERATING SURVEY IS THE WRONG INSTRUMENT. `survey`
        settles from one injection per state, which is a procedure no
        biological system performs -- nothing in cortex asks how many
        attractors it has. It also cannot answer the question it was
        being used for: chunk count came back proportional to n at a
        stable ratio (0.67, 0.80, 0.86 of the sample count across a
        fourfold spread), and "larger fields support proportionally more
        attractors" makes the SAME prediction as "the count is set by
        how many injection points I chose". One injection per state ties
        the sample count to n, so the two are indistinguishable by
        construction.

        Driving from a STREAM separates them. The stream's length is the
        sample count and is independent of n, so the two can be varied
        one at a time. And it is what actually happens: a repertoire is
        whatever the input distribution has visited, so the count is a
        property of the network AND its input, never the network alone.

        `stream` is a sequence of cues -- state indices, repeats
        expected, drawn from whatever distribution the caller is
        modelling. Returns the same shape as `survey`.
        """
        found = {}
        unstable = 0
        saved = list(self.field.a)
        saved_bias = list(self.field.bias)
        for cue in stream:
            self.reset_state()
            last = None
            held = 0
            stable = None
            for _ in range(max_fast):
                self.field.inject(cue % self.n, amount)
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
                found.setdefault(stable, []).append(cue)
        self.field.set_state(saved)
        self.field.set_bias(saved_bias)
        return found, unstable

    def survey_saturating(self, cues, max_cues=4000, max_fast=200,
                          hold=10, amount=0.2):
        """Drive until the repertoire STOPS GROWING. Sample count is an
        output, not a setting.

        WHY THE THIRD DESIGN. Every previous attempt fixed an arbitrary
        number and each one confounded the measurement differently:
        `stride=8` tied the sample count to n, so chunk count could not
        exceed n/8 and compression was reported five- to sevenfold high;
        a fixed stream length of 64 drew more UNIQUE cues at larger n
        through fewer birthday collisions, so richer sampling and a
        bigger state space moved together; fixing unique cues instead
        would then have pinned the very quantity under study. The error
        is the same in all three: a constant standing in for a stopping
        condition.

        SATURATION IS AN EVENT AND IT IS SELF-DETECTING. New cues stop
        yielding new supports once the repertoire is covered, so the run
        ends when the system says it is done rather than when a counter
        I chose runs out. The sample count then becomes a REPORTED
        quantity -- and a useful one, since how much experience a
        repertoire needs is itself worth knowing.

        PATIENCE SCALES WITH WHAT HAS BEEN FOUND, so it is not another
        constant: having found k supports, k consecutive cues bringing
        nothing new is the evidence for saturation. More repertoire
        demands more evidence, automatically. The floor of 4 is a
        minimum amount of evidence rather than a target -- without it a
        run could stop having found one support and missed one cue.

        `cues` is a callable returning the next cue, so the input is a
        SOURCE rather than a precomputed list of chosen length. Returns
        (supports, unstable, cues_consumed, saturated).
        """
        found = {}
        unstable = 0
        consumed = 0
        since_new = 0
        saturated = False
        saved = list(self.field.a)
        saved_bias = list(self.field.bias)
        while consumed < max_cues:
            cue = cues() % self.n
            consumed += 1
            self.reset_state()
            last = None
            held = 0
            stable = None
            for _ in range(max_fast):
                self.field.inject(cue, amount)
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
                continue
            if not stable:
                continue
            if stable in found:
                found[stable].append(cue)
                since_new += 1
            else:
                found[stable] = [cue]
                since_new = 0
            if since_new >= max(4, len(found)):
                saturated = True
                break
        self.field.set_state(saved)
        self.field.set_bias(saved_bias)
        return found, unstable, consumed, saturated

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

    def drive_and_stack(self, evidence, window=None, max_levels=8,
                        episodes=10, max_fast=120, stride=1,
                        outcome=1.0):
        """Build levels until a level REFUSES. Depth, compression and
        the local structure are all outputs; nothing here is a chosen
        boundary.

        `window` DEFAULTS TO NONE, and that matters. It used to default
        to a number, and every run then terminated with "fits the
        window" before reaching any limit -- so maximum depth was never
        measured, and comparing cumulative compression across runs
        compared arbitrary endpoints chosen by me. A measured 85.3x at
        n=256 against 64.0x at n=512 looked like a real reversal and was
        an artefact of where I had put the cutoff.

        The stop condition already existed and is self-detecting: a
        level refuses when its dynamics stop producing distinct chunks,
        or when its complement is absorbing. That is situation
        awareness rather than a threshold -- the level reports what it
        can do, instead of being compared against a constant.

        A window is still accepted, because a real consumer has one: a
        task's working memory is a genuine constraint rather than an
        arbitrary cutoff. It is just not the default, since measuring
        the mechanism and serving a consumer are different questions.

        A DENSITY OBSERVATION was tempting to turn into a gate. The
        coarse fan-out saturates near 18.5 while a level grows, so a
        level below roughly that size collapses to one chunk -- which
        looked like a floor worth encoding. It is NOT encoded: it is a
        property of the kernels measured so far, it would be wrong for a
        different anatomy, and the refusal already catches the case it
        would have guarded. Each level's fan-out is REPORTED instead, so
        a reader sees the local situation rather than inheriting my
        number.

        `stride` defaults to 1. At stride k, `survey` caps the chunk count
        at n/k; the previous default of 4 pinned level-0 chunks at n/4.

        Stops for a reason it names, and the last record carries it.
        """
        levels = []
        level = self
        current = list(evidence)
        for depth in range(max_levels):
            # Drive this level from below and let it learn.
            for ep in range(episodes):
                level.reset_state()
                pick = ([current[ep % len(current)]] if current
                        else [ep % level.n])
                level.task_step(evidence=pick, outcome=outcome,
                                max_fast=max_fast)
            if window is not None and level.n <= window:
                levels.append({
                    "depth": depth, "n": level.n, "chunks": None,
                    "compression": None, "fanout": None,
                    "stop": "fits the caller's window (%d <= %d)"
                            % (level.n, window),
                })
                break
            result = level.level_up(max_fast=max_fast, stride=stride)
            if not result.admissible:
                levels.append({
                    "depth": depth, "n": level.n, "chunks": None,
                    "compression": None, "fanout": None,
                    "stop": "refused: %s" % result.reason,
                })
                break
            # Observed, not asserted: the coarse kernel's own fan-out.
            nz = sum(1 for row in result.kernel for v in row
                     if v > 1e-9)
            fanout = (nz / result.chunks) if result.chunks else 0.0
            levels.append({
                "depth": depth, "n": level.n,
                "chunks": result.chunks,
                "compression": result.compression,
                "fanout": fanout,
                "stop": None,
            })
            # What this level settled on becomes the next level's input,
            # expressed in the next level's own indices.
            reps = result.representatives
            active = [i for i, _ in level.decide()]
            current = [reps.index(i) for i in active if i in reps]
            level = level.next_level(result)
        else:
            levels.append({
                "depth": max_levels, "n": level.n, "chunks": None,
                "compression": None, "fanout": None,
                "stop": "max_levels reached, still producing chunks",
            })
        return levels

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
