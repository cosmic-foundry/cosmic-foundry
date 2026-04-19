"""BoundaryCondition ABC."""

from __future__ import annotations

from cosmic_foundry.theory.function import Function


class BoundaryCondition(Function):
    """Abstract base for all boundary conditions on ∂Ω.

    A BoundaryCondition is a Function that constrains field values on the
    boundary of a Domain.  Subclasses specialize by constraint structure:
    LocalBoundaryCondition operates on a single face; NonLocalBoundaryCondition
    spans multiple faces.

    The codimension-1 invariant is enforced structurally: every face returned
    by Domain.boundary has ndim = parent.ndim - 1.

    execute is left fully abstract here; concrete subclasses in computation/
    supply the JAX-backed implementation with a typed signature of the form
    execute(domain, face, field_data).
    """


__all__ = ["BoundaryCondition"]
