"""LocalBoundaryCondition ABC."""

from __future__ import annotations

from abc import abstractmethod
from typing import TypeVar

from cosmic_foundry.theory.boundary_condition import BoundaryCondition
from cosmic_foundry.theory.field import Field

D = TypeVar("D")  # Domain
C = TypeVar("C")  # Codomain


class LocalBoundaryCondition(BoundaryCondition[D, C]):
    """A boundary condition that constrains a single face of ∂Ω.

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


__all__ = ["LocalBoundaryCondition"]
