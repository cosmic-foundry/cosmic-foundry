"""Discretization ABC."""

from __future__ import annotations

from abc import ABC, abstractmethod

from cosmic_foundry.theory.discrete.discrete_boundary_condition import (
    DiscreteBoundaryCondition,
)
from cosmic_foundry.theory.discrete.discrete_operator import DiscreteOperator
from cosmic_foundry.theory.discrete.mesh import Mesh


class Discretization(ABC):
    """Encapsulates a discrete scheme on a mesh.

    A Discretization holds the scheme choice — reconstruction, numerical
    flux, quadrature — for a particular mesh and approximation order.
    Calling it produces the DiscreteOperator Lₕ that makes the commutation
    diagram

        Lₕ ∘ Rₕ ≈ Rₕ ∘ L   (up to O(hᵖ))

    hold, where p is the approximation order.

    Required:
        __call__ — produce the DiscreteOperator (signature defined by subclass)

    Concrete:
        mesh               — the mesh on which the scheme is defined
        boundary_condition — the DiscreteBoundaryCondition on ∂Ω (None if not yet set)
    """

    def __init__(
        self,
        mesh: Mesh,
        boundary_condition: DiscreteBoundaryCondition | None = None,
    ) -> None:
        self._mesh = mesh
        self._boundary_condition = boundary_condition

    @property
    def mesh(self) -> Mesh:
        """The mesh on which the scheme is defined."""
        return self._mesh

    @property
    def boundary_condition(self) -> DiscreteBoundaryCondition | None:
        """The discrete boundary condition on ∂Ω."""
        return self._boundary_condition

    @abstractmethod
    def __call__(self) -> DiscreteOperator:
        """Produce the assembled DiscreteOperator."""


__all__ = ["Discretization"]
