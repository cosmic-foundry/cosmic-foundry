"""NonLocalBoundaryCondition ABC."""

from __future__ import annotations

from cosmic_foundry.theory.boundary_condition import BoundaryCondition


class NonLocalBoundaryCondition(BoundaryCondition):
    """A boundary condition whose constraint spans more than one location.

    A local boundary condition can be evaluated from field data at a single
    face in isolation.  A non-local boundary condition cannot — it requires
    data from multiple locations, which may be boundary faces, the whole
    boundary, or interior points depending on the concrete type.

    This class makes no claim about the form of the non-locality.  Concrete
    subclasses declare whatever geometric references they need:
    FaceIdentification carries a pair of boundary faces; a Dirichlet-to-Neumann
    map carries the full boundary; a radiation condition may reference interior
    data.
    """


__all__ = ["NonLocalBoundaryCondition"]
