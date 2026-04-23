"""LocalBoundaryCondition ABC."""

from __future__ import annotations

from abc import abstractmethod

from cosmic_foundry.theory.continuous.boundary_condition import BoundaryCondition
from cosmic_foundry.theory.continuous.field import Field
from cosmic_foundry.theory.continuous.manifold import Manifold


class LocalBoundaryCondition(BoundaryCondition):
    """A boundary condition that constrains a single face of ∂M.

    Represents the Robin family: α·f + β·∂f/∂n = g, where f is the field
    value at the face, ∂f/∂n is the outward normal derivative, and g is
    the prescribed data.  Special cases:

        Dirichlet — alpha=1, beta=0: prescribed field value
        Neumann   — alpha=0, beta=1: prescribed normal derivative
        Robin     — alpha≠0, beta≠0: linear combination

    Required:
        alpha      — coefficient of the field value at the face
        beta       — coefficient of the outward normal derivative
        constraint — the prescribed data g; a Field on the boundary face

    Derived:
        support    — the manifold on which this BC is enforced; equals
                     constraint.manifold
    """

    @property
    @abstractmethod
    def alpha(self) -> float:
        """Coefficient of the field value (f) in α·f + β·∂f/∂n = g."""

    @property
    @abstractmethod
    def beta(self) -> float:
        """Coefficient of the normal derivative (∂f/∂n) in α·f + β·∂f/∂n = g."""

    @property
    @abstractmethod
    def constraint(self) -> Field:
        """The prescribed boundary data g in α·f + β·∂f/∂n = g."""

    @property
    def support(self) -> Manifold:
        """The manifold on which this BC is enforced; derived as constraint.manifold."""
        return self.constraint.manifold


__all__ = ["LocalBoundaryCondition"]
