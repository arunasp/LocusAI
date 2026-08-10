"""Values that do not decide.

`Cascade` scales its commitment threshold and its convergence requirement by
an action's reversibility, but nothing produces that number -- it arrives as
a caller-supplied argument. This is where it comes from.

The constitution never chooses an action. It states what an action costs,
and the cascade already respects cost. A grave act needs more evidence; that
is the whole mechanism. Nothing here inspects the reasoning that proposed
the act, only the act itself, because a declared class is checkable and a
justification is not.

Separately and much more rarely, a small categorical set holds regardless of
evidence. That set is not a threshold: reversibility only *scales* what the
cascade demands, so sufficient evidence eventually clears anything scaled.
The remainder is what no amount of evidence clears. It is deliberately tiny,
and its firing rate is a health metric -- a rising rate means the costs above
are failing, not that the guard is working.

See doc/core/CONSTITUTION.md for why values enter as cost rather than as a
filter on candidates, and why the veto sits after the cascade rather than
inside it.
"""

import types

# An act whose class is not named carries the highest cost, not the lowest.
# Silence in a value system is not permission: an unclassified act is one
# nobody has reasoned about, which is precisely when caution is cheapest to
# apply and most likely to be wanted. The alternative -- unknown means free --
# makes the constitution weakest exactly where its coverage is thinnest.
UNCLASSIFIED_REVERSIBILITY = 0.05


class Constitution:
    """Supplies an act's cost, and refuses a small categorical set.

    Costs and refusals are fixed at construction. There is no method to add,
    remove or adjust either, and the stored mappings are read-only proxies:
    the layer this binds cannot rewrite what binds it, which is the property
    that makes it a constraint rather than a preference. Changing values
    means building a new Constitution, which is a deliberate act at a level
    above the one being constrained.
    """

    def __init__(self, costs=None, forbidden=()):
        costs = dict(costs or {})
        for name, value in costs.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    "reversibility for %r must lie in [0, 1]" % (name,))
        self._costs = types.MappingProxyType(costs)
        self._forbidden = frozenset(forbidden)
        overlap = self._forbidden & set(costs)
        if overlap:
            # A forbidden act with a cost invites the reading that enough
            # evidence buys it. It cannot, so the two must not both apply.
            raise ValueError(
                "forbidden acts must not carry a cost: %s"
                % ", ".join(sorted(overlap)))

    @property
    def costs(self):
        """Read-only view of the declared reversibility per action class."""
        return self._costs

    @property
    def forbidden(self):
        """Read-only view of the categorical set."""
        return self._forbidden

    def reversibility(self, action_class):
        """Cost of an act, as the cascade wants it: 1.0 undoable, 0 not.

        An unnamed class returns UNCLASSIFIED_REVERSIBILITY rather than 1.0.
        """
        return self._costs.get(action_class, UNCLASSIFIED_REVERSIBILITY)

    def permits(self, action_class):
        """False only for the categorical set. Evidence does not enter."""
        return action_class not in self._forbidden
