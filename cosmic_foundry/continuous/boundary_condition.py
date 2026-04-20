"""BoundaryCondition ABC."""

from __future__ import annotations

from typing import TypeVar

from cosmic_foundry.foundation.function import Function

D = TypeVar("D")  # Domain (boundary constraint)
C = TypeVar("C")  # Codomain (constraint value)


class BoundaryCondition(Function[D, C]):
    """Abstract base for all boundary conditions on ∂Ω.

    A BoundaryCondition is a Function that constrains field values on the
    boundary of a Domain.  Subclasses specialize by constraint structure:
    LocalBoundaryCondition operates on a single face; NonLocalBoundaryCondition
    spans multiple faces.

    The codimension-1 invariant is enforced structurally: every face returned
    by Domain.boundary has ndim = parent.ndim - 1.

    __call__ is left fully abstract here; concrete subclasses in computation/
    supply the JAX-backed implementation with a typed signature of the form
    __call__(domain, face, field_data).
    """


__all__ = ["BoundaryCondition"]
