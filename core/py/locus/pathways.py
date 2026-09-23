"""Pathway dispatch.

Three structurally distinct routes to an action, not one network blending
everything into a shared representation:

  instinct     a discrete releaser table. An exact cue fires an all-or-nothing
               program. No scoring, no probability mass, no blending -- the
               property that separates a fixed action pattern from a heavily
               weighted tendency.
  procedural   a consolidated correction, cheap in proportion to how often it
               has already worked.
  declarative  retrieval and fresh judgement, flexible and expensive, and the
               only route whose reads destabilise what they touch.

The fast route acts before the slow route finishes; the slow route may then
revise what the fast route already did. Whether it is allowed to act at all,
rather than merely orient, is governed by the commitment cascade.
"""

import dataclasses

from locus.commit import Cascade, Stage
from locus.decisions import DecisionTrace
from locus.store import Pathway

RELEASER_EVIDENCE = 0.6


@dataclasses.dataclass
class Outcome:
    """What was done, by which route, and at what deliberation cost."""

    action: object = None
    source: str = "none"
    stage: Stage = Stage.HOLD
    cost: float = 0.0
    revision: object = None
    revised: bool = False
    vetoed: bool = False


class Dispatcher:
    """Route a cue through the three pathways in structural order."""

    def __init__(self, store, consolidator, cascade=None,
                 releaser_evidence=RELEASER_EVIDENCE, constitution=None,
                 trace=None):
        self.store = store
        self.consolidator = consolidator
        self.cascade = cascade or Cascade()
        self.releaser_evidence = releaser_evidence
        # The constitution supplies an act's cost from its declared class
        # and refuses the categorical set. It is held, never built here:
        # a layer that constructs what binds it is not bound by it.
        self.constitution = constitution
        self.trace = trace if trace is not None else DecisionTrace()
        self._releasers = {}
        self._classes = {}

    def add_releaser(self, cue, key, action, reversibility=None,
                     data=None, action_class=None):
        """Install a discrete trigger. Pinned resident: it can never miss.

        With a constitution, an act's cost comes from its DECLARED CLASS
        rather than from the caller: a class is checkable, a caller's
        claim about its own act is not. An unnamed class is the most
        expensive, not the cheapest.
        """
        if reversibility is None:
            reversibility = (self.constitution.reversibility(action_class)
                             if self.constitution is not None else 1.0)
        payload = data if data is not None else str(action)
        self.store.put(
            key, payload, pathway=Pathway.INSTINCT, salience=1.0
        )
        self._releasers[cue] = (key, action, reversibility)
        self._classes[cue] = action_class

    def fires(self, cue):
        """Exact membership. Discreteness lives here, and only here."""
        return cue in self._releasers

    def _resolve(self, cue):
        """Store key backing a cue, or None if the cue is not trace-backed.

        A releaser's trace lives under its registered key, not under the cue
        that fires it; the two key spaces are deliberately separate.
        """
        if cue in self._releasers:
            return self._releasers[cue][0]
        return cue if isinstance(cue, int) else None

    def act(self, cue, judge=None, signals=1, action_class=None):
        """Dispatch one cue.

        judge, when supplied, is the slow pathway: it receives the trace
        payload and returns (evidence, action). It runs after the fast route
        has already committed, and its result may overturn it.
        """
        outcome = Outcome()
        key = self._resolve(cue)
        cls = action_class if action_class is not None \
            else self._classes.get(cue)

        # The veto sits BEFORE any route runs and after no amount of
        # evidence: reversibility only scales what the cascade demands,
        # so a scaled cost is eventually payable and this is not. It is
        # recorded and excluded from learning -- nothing below is
        # reached, so no pathway is reinforced by the refusal.
        if self.constitution is not None and not self.constitution.permits(
                cls):
            outcome.vetoed = True
            self.trace.record(
                cue=cue, action_class=cls, reversibility=0.0,
                evidence=0.0, signals=signals, stage=int(Stage.HOLD),
                vetoed=True, source="veto")
            return outcome

        if self.fires(cue):
            _key, action, reversibility = self._releasers[cue]
            outcome.action = action
            outcome.source = "instinct"
            outcome.cost = 0.0
            outcome.stage = self.cascade.stage(
                self.releaser_evidence, reversibility, signals
            )
            self.trace.record(
                cue=cue, action_class=cls, reversibility=reversibility,
                evidence=self.releaser_evidence, signals=signals,
                stage=int(outcome.stage), source="instinct")
        elif key is not None and self.consolidator.is_habit(key):
            lease = self.store.lookup(key)
            payload = lease.data if lease is not None else b""
            if lease is not None:
                lease.release()
            self.consolidator.applied(key, surprise=0.0)
            outcome.action = payload
            outcome.source = "procedural"
            outcome.cost = self.consolidator.cost(key)
            outcome.stage = Stage.COMMIT

        if judge is None or key is None:
            return outcome

        if outcome.source == "none":
            # Declarative recall: the only route whose read destabilises.
            lease = self.store.retrieve(key)
            if lease is None:
                return outcome
            try:
                evidence, revised_action = judge(lease.data)
            finally:
                lease.release()
            self.store.restabilize(key, revised_action)
            outcome.action = revised_action
            outcome.source = "declarative"
            outcome.cost = self.consolidator.cost(key)
            outcome.stage = self.cascade.stage(evidence, 1.0, signals)
            return outcome

        # An action was already taken. Deliberation may overturn it, but it
        # reads without destabilising and never rewrites a pinned releaser.
        lease = self.store.lookup(key)
        if lease is None:
            return outcome
        try:
            _evidence, revised_action = judge(lease.data)
        finally:
            lease.release()
        if revised_action != outcome.action:
            outcome.revision = revised_action
            outcome.revised = True
        return outcome
