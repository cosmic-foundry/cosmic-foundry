"""Finite directed transition systems."""

from __future__ import annotations

from dataclasses import dataclass

from cosmic_foundry.theory.foundation.indexed_set import IndexedSet


@dataclass(frozen=True)
class FiniteStateTransitionSystem(IndexedSet):
    """Finite states connected by directed unit-transfer transitions.

    A transition ``i -> j`` removes one unit from state ``i`` and adds one unit
    to state ``j``.  The resulting stoichiometry matrix therefore has one
    ``-1`` and one ``+1`` in each column, so the all-ones row vector is a
    conserved linear form.
    """

    state_count: int
    transitions: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if self.state_count <= 0:
            raise ValueError("finite transition systems require at least one state")
        normalized = tuple((int(src), int(dst)) for src, dst in self.transitions)
        for src, dst in normalized:
            if src == dst:
                raise ValueError("unit-transfer transitions require distinct states")
            if not (0 <= src < self.state_count and 0 <= dst < self.state_count):
                raise ValueError("transition endpoint outside finite state set")
        object.__setattr__(self, "transitions", normalized)

    @classmethod
    def chain(cls, state_count: int) -> FiniteStateTransitionSystem:
        """Return the directed chain ``0 -> 1 -> ... -> n - 1``."""
        return cls(
            state_count,
            tuple((state, state + 1) for state in range(state_count - 1)),
        )

    @property
    def shape(self) -> tuple[int, ...]:
        """Finite state set shape."""
        return (self.state_count,)

    def intersect(self, other: IndexedSet) -> FiniteStateTransitionSystem | None:
        """Return the common finite prefix state set, preserving common edges."""
        if not isinstance(other, FiniteStateTransitionSystem):
            return None
        state_count = min(self.state_count, other.state_count)
        if state_count <= 0:
            return None
        other_edges = set(other.transitions)
        transitions = tuple(
            edge
            for edge in self.transitions
            if edge in other_edges and edge[0] < state_count and edge[1] < state_count
        )
        return FiniteStateTransitionSystem(state_count, transitions)

    @property
    def transition_count(self) -> int:
        """Number of directed transitions."""
        return len(self.transitions)

    def conserved_total_form(self) -> tuple[int, ...]:
        """Return the conserved all-ones linear form."""
        return (1,) * self.state_count

    def stoichiometry_matrix(self) -> tuple[tuple[int, ...], ...]:
        """Return the state-by-transition unit-transfer stoichiometry matrix."""
        rows = [
            [0 for _transition in self.transitions]
            for _state in range(self.state_count)
        ]
        for column, (src, dst) in enumerate(self.transitions):
            rows[src][column] = -1
            rows[dst][column] = 1
        return tuple(tuple(row) for row in rows)


__all__ = ["FiniteStateTransitionSystem"]
