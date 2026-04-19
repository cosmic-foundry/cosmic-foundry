"""NonLocalBoundaryCondition ABC."""

from __future__ import annotations

from abc import abstractmethod

from cosmic_foundry.theory.boundary_condition import BoundaryCondition
from cosmic_foundry.theory.manifold_with_boundary import ManifoldWithBoundary


class NonLocalBoundaryCondition(BoundaryCondition):
    """A boundary condition whose constraint spans multiple faces of ∂Ω.

    NonLocalBoundaryCondition reads field data from more than one face and
    cannot be evaluated from a single face in isolation.  The canonical
    concrete subclass is FaceIdentification (periodic BC): two faces, identity
    map.  Further variants include anti-periodic (negation map) and
    Bloch/Floquet (phase map, used in photonics and solid-state).

    Required:
        sources — the faces involved in the constraint, each a member of
                  Domain.boundary (and therefore of dimension ndim-1)
    """

    @property
    @abstractmethod
    def sources(self) -> tuple[ManifoldWithBoundary, ...]:
        """The boundary faces this condition reads from."""


__all__ = ["NonLocalBoundaryCondition"]
