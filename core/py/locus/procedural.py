"""Procedural consolidation: repetition buys cheapness.

Layer 1 of the split. Layer 2 records a correction after one occurrence;
this layer counts how often that record has been successfully re-applied
and discounts the cost of applying it again. Promotion into the pinned
tier is delegated to the store, because repetition-count promotion and
tier promotion are the same mechanism.
"""

DELIBERATE_COST = 1.0
FLOOR_COST = 0.05


class Consolidator:
    """Track successful re-application and report deliberation cost."""

    def __init__(self, store, discount=0.5, floor=FLOOR_COST):
        if not 0.0 < discount < 1.0:
            raise ValueError("discount must lie in (0, 1)")
        self.store = store
        self.discount = discount
        self.floor = floor

    def cost(self, key):
        """Cost of applying a stored correction again.

        Full deliberation on first use, decaying geometrically with each
        confirmed re-application down to a floor. A pinned trace is at the
        floor by definition: it no longer competes for residency.
        """
        reps = self.store.reps(key)
        if reps is None or reps < 0:
            return DELIBERATE_COST
        if self.store.pinned(key):
            return self.floor
        return max(self.floor, DELIBERATE_COST * (self.discount ** reps))

    def applied(self, key, surprise=0.0):
        """Record one re-application, carrying its prediction error.

        Repetition alone does not build a habit: the store only counts a
        re-application toward promotion when its outcome had stopped being
        surprising. Pass |delta| for the step; the default assumes a fully
        predicted outcome.
        """
        return self.store.reinforce(key, surprise)

    def is_habit(self, key):
        return self.store.pinned(key)
