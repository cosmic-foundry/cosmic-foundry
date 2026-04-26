"""BoundaryCondition ABC."""

from __future__ import annotations

from cosmic_foundry.theory.continuous.constraint import Constraint


class BoundaryCondition(Constraint):
    """Abstract base for all boundary conditions on ∂M.

    A BoundaryCondition is a Constraint whose support is the boundary of a
    manifold.  Subclasses specialize by constraint structure:
    LocalBoundaryCondition operates on a single face; NonLocalBoundaryCondition
    spans multiple faces.
    """


__all__ = ["BoundaryCondition"]
