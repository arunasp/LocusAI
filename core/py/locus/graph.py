"""Associative graph and spreading activation.

Retrieval mechanism for the declarative pathway: activation spreads
mechanically across weighted links with per-hop decay, surfacing traces
nobody queried for. Graph traversal with decay, not similarity search.
"""


class AssociativeGraph:
    """Weighted, undirected-by-default associative link structure."""

    def __init__(self, decay=0.5, floor=0.01, depth=3):
        if not 0.0 < decay < 1.0:
            raise ValueError("decay must lie in (0, 1)")
        self.decay = decay
        self.floor = floor
        self.depth = depth
        self._edges = {}

    def link(self, a, b, weight=1.0, symmetric=True):
        """Associate two traces. Repeated links strengthen, capped at 1."""
        if not 0.0 < weight <= 1.0:
            raise ValueError("weight must lie in (0, 1]")
        self._edges.setdefault(a, {})
        self._edges.setdefault(b, {})
        self._edges[a][b] = min(1.0, self._edges[a].get(b, 0.0) + weight)
        if symmetric:
            self._edges[b][a] = min(1.0, self._edges[b].get(a, 0.0) + weight)

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
        return activation

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
