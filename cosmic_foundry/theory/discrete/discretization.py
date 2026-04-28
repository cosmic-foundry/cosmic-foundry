"""Discretization ABC."""

from __future__ import annotations

from typing import TypeVar

from cosmic_foundry.theory.discrete.discrete_boundary_condition import (
    DiscreteBoundaryCondition,
)
from cosmic_foundry.theory.discrete.discrete_operator import DiscreteOperator
from cosmic_foundry.theory.discrete.mesh import Mesh

_V = TypeVar("_V")


class Discretization(DiscreteOperator[_V]):
    """A DiscreteOperator defined on a specific mesh with a chosen scheme.

    A Discretization IS the discrete operator Lₕ that makes the commutation
    diagram

        Lₕ ∘ Rₕ ≈ Rₕ ∘ L   (up to O(hᵖ))

    hold at convergence order p.  Each mesh — including each AMR level — has
    its own Discretization instance; there is no shared "scheme object" that
    produces operators for varying meshes.

    Required (from DiscreteOperator):
        order               — integer convergence order
        continuous_operator — the DifferentialOperator this approximates
        __call__            — apply Lₕ: DiscreteField → DiscreteField

    Concrete:
        mesh               — the mesh on which the scheme is defined
        boundary_condition — the DiscreteBoundaryCondition on ∂Ω (None if not set)
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


__all__ = ["Discretization"]
