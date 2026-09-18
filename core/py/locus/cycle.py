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
        self.field = Field(n, kernel, rho=rho, g=g, beta=beta, leak=leak)
        self.plasticity = Plasticity(trace_decay=trace_decay, rate=rate)
        self.steps = 0

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
        """
        fast, settled = self.settle(evidence=evidence, max_fast=max_fast)
        active = self.decide()
        tags, touched = self.learn(outcome)
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
