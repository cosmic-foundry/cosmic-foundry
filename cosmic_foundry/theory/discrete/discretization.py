"""Discretization ABC."""

from __future__ import annotations

from typing import TypeVar

from cosmic_foundry.theory.discrete.discrete_boundary_condition import (
    DiscreteBoundaryCondition,
    ZeroGhostCells,
)
from cosmic_foundry.theory.discrete.discrete_operator import DiscreteOperator

_V = TypeVar("_V")


class Discretization(DiscreteOperator[_V]):
    """A DiscreteOperator defined by a scheme and boundary condition.

    A Discretization IS the discrete operator Lₕ that makes the commutation
    diagram

        Lₕ ∘ Rₕ ≈ Rₕ ∘ L   (up to O(hᵖ))

    hold at convergence order p.  The active mesh is always derived from the
    input field in __call__ (U.mesh), not stored on the discretization; the
    same scheme object can therefore be probed with symbolic-spacing fields
    by the order-verification infrastructure.

    Required (from DiscreteOperator):
        order               — integer convergence order
        continuous_operator — the DifferentialOperator this approximates
        __call__            — apply Lₕ: DiscreteField → DiscreteField

    Concrete:
        boundary_condition — the DiscreteBoundaryCondition on ∂Ω.  Defaults to
                             ZeroGhostCells() when not supplied (absorbing /
                             no-BC behavior).
    """

    def __init__(
        self,
        boundary_condition: DiscreteBoundaryCondition | None = None,
    ) -> None:
        self._boundary_condition: DiscreteBoundaryCondition = (
            boundary_condition if boundary_condition is not None else ZeroGhostCells()
        )

    @property
    def boundary_condition(self) -> DiscreteBoundaryCondition:
        """The discrete boundary condition on ∂Ω."""
        return self._boundary_condition


__all__ = ["Discretization"]
