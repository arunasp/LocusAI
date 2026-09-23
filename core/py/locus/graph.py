"""Associative graph and spreading activation.

Retrieval mechanism for the declarative pathway: activation spreads
mechanically across weighted links with per-hop decay, surfacing traces
nobody queried for. Graph traversal with decay, not similarity search.

This is also, by function, a RE-DESCRIPTION ENGINE
(doc/core/CONSTITUTION.md, Known limits): it finds routes between traces
without intending to, and a route is how a costed effect gets reached by
an uncosted path. So linking is constrained rather than public and free:

- a link between INCOMPATIBLE CLASSES is refused outright, because it
  would make one directly reachable from the other;
- an incompatible pair reached INDIRECTLY, over two or more permitted
  links, cannot be prevented by any rule on single links, so it is
  RECORDED instead. The constitution is explicit that prevention here is
  partial and the honest goal is to make circumvention visible.

The class map and the conflict relation are supplied at construction and
held read-only; nothing here can widen what it is allowed to connect.
"""

import types


class AssociativeGraph:
    """Weighted, undirected-by-default associative link structure."""

    def __init__(self, decay=0.5, floor=0.01, depth=3, classes=None,
                 conflicts=None, trace=None):
        if not 0.0 < decay < 1.0:
            raise ValueError("decay must lie in (0, 1)")
        self.decay = decay
        self.floor = floor
        self.depth = depth
        self._edges = {}
        # key -> class, and class -> the classes it may not join. Both are
        # read-only proxies: the layer that links does not get to edit
        # what it may link.
        self._classes = types.MappingProxyType(dict(classes or {}))
        self._conflicts = types.MappingProxyType(
            {k: frozenset(v) for k, v in dict(conflicts or {}).items()})
        self.trace = trace

    @property
    def classes(self):
        """Read-only view of the class of each key."""
        return self._classes

    @property
    def conflicts(self):
        """Read-only view of the incompatibility relation."""
        return self._conflicts

    def incompatible(self, a, b):
        """Do these two keys hold classes that may not be joined?

        Symmetric whichever direction was declared: reachability has no
        direction, so neither does the refusal.
        """
        ca, cb = self._classes.get(a), self._classes.get(b)
        if ca is None or cb is None:
            return False
        return (cb in self._conflicts.get(ca, ())
                or ca in self._conflicts.get(cb, ()))

    def link(self, a, b, weight=1.0, symmetric=True):
        """Associate two traces. Repeated links strengthen, capped at 1.

        Returns True when the link was made, False when it was refused
        for incompatibility. A refusal is recorded where a trace exists
        and NEVER raises: a refused association is a normal event in a
        constrained system, not an error in the caller.
        """
        if not 0.0 < weight <= 1.0:
            raise ValueError("weight must lie in (0, 1]")
        if self.incompatible(a, b):
            if self.trace is not None:
                self.trace.record(
                    cue=(a, b),
                    action_class=(self._classes.get(a),
                                  self._classes.get(b)),
                    reversibility=0.0, evidence=weight, signals=1,
                    stage=0, vetoed=True, source="link-refused")
            return False
        self._edges.setdefault(a, {})
        self._edges.setdefault(b, {})
        self._edges[a][b] = min(1.0, self._edges[a].get(b, 0.0) + weight)
        if symmetric:
            self._edges[b][a] = min(1.0, self._edges[b].get(a, 0.0) + weight)
        return True

    def neighbours(self, key):
        return dict(self._edges.get(key, {}))

    def spread(self, origins, depth=None, decay=None):
        """Return accumulated activation reachable from origins.

        Origins may be a mapping of key -> energy or a bare iterable of keys.
        """
        if not isinstance(origins, dict):
            origins = {key: 1.0 for key in origins}
        depth = self.depth if depth is None else depth
        decay = self.decay if decay is None else decay

        activation = dict(origins)
        frontier = dict(origins)
        for _ in range(max(0, depth)):
            nxt = {}
            for node, energy in frontier.items():
                for peer, weight in self._edges.get(node, {}).items():
                    delta = energy * weight * decay
                    if delta < self.floor:
                        continue
                    nxt[peer] = nxt.get(peer, 0.0) + delta
            if not nxt:
                break
            for key, value in nxt.items():
                activation[key] = activation.get(key, 0.0) + value
            frontier = nxt
        self._note_indirect(activation)
        return activation

    def _note_indirect(self, activation):
        """Record an incompatible pair reached over permitted links.

        No rule on single links can stop this: each hop is allowed and
        the pair is only incompatible end to end. Detection, not
        prevention (doc/core/CONSTITUTION.md, Known limits).
        """
        if self.trace is None or not self._conflicts:
            return
        live = [k for k, v in activation.items()
                if v >= self.floor and k in self._classes]
        for i, a in enumerate(live):
            for b in live[i + 1:]:
                if self.incompatible(a, b):
                    self.trace.record(
                        cue=(a, b),
                        action_class=(self._classes[a], self._classes[b]),
                        reversibility=0.0,
                        evidence=min(activation[a], activation[b]),
                        signals=1, stage=0, vetoed=False,
                        source="reached-indirectly")

    def prefetch_set(self, origins, limit=8):
        """Keys worth warming, strongest first, excluding the origins.

        Advisory by construction: the caller may ignore the result entirely
        without affecting correctness.
        """
        seeds = set(origins if not isinstance(origins, dict) else origins)
        ranked = sorted(
            ((k, v) for k, v in self.spread(origins).items()
             if k not in seeds),
            key=lambda item: item[1],
            reverse=True,
        )
        return [key for key, _ in ranked[:limit]]
