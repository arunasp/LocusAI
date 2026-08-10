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

    def __init__(self, threshold=0.35, salience_gain=1.0):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must lie in [0, 1]")
        self.threshold = threshold
        self.salience_gain = salience_gain
        self._known = {}

    def novelty(self, features):
        """Prediction error: 1.0 when nothing resembles these features."""
        features = frozenset(features)
        if not self._known:
            return 1.0
        return 1.0 - max(
            _jaccard(features, seen) for seen in self._known.values()
        )

    def admit(self, key, features):
        """Decide whether to encode. Returns (encode, salience)."""
        error = self.novelty(features)
        if error < self.threshold:
            return False, 0.0
        self._known[key] = frozenset(features)
        return True, error * self.salience_gain

    def forget(self, key):
        self._known.pop(key, None)

    def __len__(self):
        return len(self._known)
