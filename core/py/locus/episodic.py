"""Novelty-gated episodic encoding.

Layer 2 of the consolidation split: encoding fires on prediction error, not
on repetition. One occurrence is enough; a familiar event writes nothing.
"""


def _jaccard(a, b):
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


class NoveltyGate:
    """Gate encoding on how unlike anything already stored an event is."""

    def __init__(self, threshold=0.35, salience_gain=1.0, store=None):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must lie in [0, 1]")
        self.threshold = threshold
        self.salience_gain = salience_gain
        # WHAT THE GATE COMPARES AGAINST is what is still remembered,
        # not everything ever admitted. Given a store, a key whose trace
        # the store no longer holds stops counting as known, so an event
        # can become novel again once its trace is gone -- which is the
        # point of a novelty gate sitting in front of a finite memory.
        # Without a store the gate keeps everything, as before.
        self._store = store
        self._known = {}

    def novelty(self, features):
        """Prediction error: 1.0 when nothing resembles these features.

        Forgetting RIDES THIS COMPARISON rather than adding a sweep of
        its own: the loop already visits every known key, so dropping
        the dead ones costs nothing extra and happens exactly when the
        set is used.
        """
        features = frozenset(features)
        closest = 0.0
        dead = []
        for key, seen in self._known.items():
            if self._store is not None and self._store.tier(key) is None:
                dead.append(key)
                continue
            closest = max(closest, _jaccard(features, seen))
        for key in dead:
            del self._known[key]
        if not self._known:
            return 1.0
        return 1.0 - closest

    def admit(self, key, features):
        """Decide whether to encode. Returns (encode, salience)."""
        error = self.novelty(features)
        if error < self.threshold:
            return False, 0.0
        self._known[key] = frozenset(features)
        return True, error * self.salience_gain

    def __len__(self):
        return len(self._known)
