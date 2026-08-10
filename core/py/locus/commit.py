"""Reversibility-scaled commitment.

Signal Detection Theory puts the optimal criterion at the relative cost of
the two error types, not at a fixed point. An irreversible action makes a
false positive catastrophic and a false negative merely slow, so the
threshold for committing rises while the threshold for orienting does not.

Staged, as a predator strike is staged: orienting and preparing stay cheap
and abortable; only the terminal stage demands convergent evidence.
"""

import enum


class Stage(enum.IntEnum):
    HOLD = 0
    ORIENT = 1
    PREPARE = 2
    COMMIT = 3


REVERSIBILITY_FLOOR = 0.05


class Cascade:
    """Decide how far along the action cascade evidence justifies going."""

    def __init__(self, orient=0.2, prepare=0.5, commit=0.8,
                 convergence=2):
        if not orient <= prepare <= commit:
            raise ValueError("stage thresholds must be non-decreasing")
        self.orient = orient
        self.prepare = prepare
        self.commit = commit
        self.convergence = convergence

    def commit_threshold(self, reversibility):
        """Evidence needed to commit, scaled by how undoable the act is.

        reversibility is 1.0 for a fully undoable action and approaches 0
        for one that cannot be taken back.
        """
        if not 0.0 <= reversibility <= 1.0:
            raise ValueError("reversibility must lie in [0, 1]")
        return self.commit / max(reversibility, REVERSIBILITY_FLOOR)

    def required_signals(self, reversibility):
        """Independent confirmations the terminal stage demands."""
        if not 0.0 <= reversibility <= 1.0:
            raise ValueError("reversibility must lie in [0, 1]")
        scale = 1.0 / max(reversibility, REVERSIBILITY_FLOOR)
        return max(1, int(round(self.convergence * scale)))

    def stage(self, evidence, reversibility=1.0, signals=1):
        """Highest stage this evidence justifies.

        Orienting and preparing are deliberately insensitive to
        reversibility: getting ready to act costs nothing to undo.
        """
        if evidence < self.orient:
            return Stage.HOLD
        if evidence < self.prepare:
            return Stage.ORIENT
        if (evidence >= self.commit_threshold(reversibility)
                and signals >= self.required_signals(reversibility)):
            return Stage.COMMIT
        return Stage.PREPARE
