"""NonLocalBoundaryCondition ABC."""

from __future__ import annotations

from cosmic_foundry.theory.continuous.boundary_condition import BoundaryCondition


class NonLocalBoundaryCondition(BoundaryCondition):
    """A boundary condition whose constraint is non-local.

    A local boundary condition depends only on field values in an infinitesimal
    neighborhood of the boundary point being constrained (the value and its
    normal derivative at that point).  A non-local boundary condition depends
    on field values elsewhere — at a distant boundary face, over an integral of
    the whole boundary, at interior points, or at infinity.

    This class makes no claim about the form of the non-locality.  Concrete
    subclasses declare whatever geometric references they need:
    FaceIdentification carries a pair of boundary faces; a Dirichlet-to-Neumann
    map carries the full boundary; a radiation condition may reference interior
    data.
    """


__all__ = ["NonLocalBoundaryCondition"]
