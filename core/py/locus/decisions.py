"""What was decided, kept separately from what is learned.

doc/core/CONSTITUTION.md requires two things of a veto that this module
supplies. It must be recorded, so circumvention is visible: the honest
goal is detection, since any rule keyed on a description of an act can be
reached by a differently-framed route. And it must NOT feed plasticity --
a veto reported as prediction error teaches the system to avoid
TRIGGERING the veto rather than to comply, and improving at that looks
exactly like compliance.

So the trace is an audit surface and nothing reads it to learn. Records
are append-only and the sequence handed out is a copy: the layer being
audited cannot edit its own record.
"""

import dataclasses
import types


@dataclasses.dataclass(frozen=True)
class Decision:
    """One dispatch, as the audit sees it."""

    cue: object
    action_class: object
    reversibility: float
    evidence: float
    signals: int
    stage: int
    vetoed: bool = False
    source: str = "none"

    def __str__(self):
        return ("%s cue=%r class=%r rev=%.3f evidence=%.3f signals=%d "
                "stage=%d%s"
                % (self.source, self.cue, self.action_class,
                   self.reversibility, self.evidence, self.signals,
                   self.stage, " VETOED" if self.vetoed else ""))


class DecisionTrace:
    """Append-only record of decisions, including refused ones."""

    def __init__(self):
        self._records = []

    def record(self, **fields):
        self._records.append(Decision(**fields))
        return self._records[-1]

    @property
    def records(self):
        """A copy: an audit the audited party can edit is not an audit."""
        return tuple(self._records)

    def __len__(self):
        return len(self._records)

    def vetoes(self):
        """The refused decisions, whose RATE is the health metric.

        A rising rate means the costs are failing to do their work, not
        that the guard is working well (doc/core/CONSTITUTION.md).
        """
        return tuple(r for r in self._records if r.vetoed)

    def rate(self):
        """Share of decisions refused outright."""
        return len(self.vetoes()) / len(self._records) if self._records \
            else 0.0

    def render(self):
        """The trace as lines, for a person rather than a metric."""
        return tuple(str(r) for r in self._records)


EMPTY = types.MappingProxyType({})
