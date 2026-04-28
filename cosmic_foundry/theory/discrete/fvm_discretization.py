"""FVMDiscretization: finite-volume DiscreteOperator on a CartesianMesh."""

from __future__ import annotations

from typing import Any, cast

import sympy

from cosmic_foundry.geometry.cartesian_exterior_derivative import (
    CartesianExteriorDerivative,
)
from cosmic_foundry.geometry.cartesian_mesh import CartesianMesh
from cosmic_foundry.theory.continuous.differential_operator import (
    DifferentialOperator,
    DivergenceComposition,
)
from cosmic_foundry.theory.discrete.discrete_boundary_condition import (
    DiscreteBoundaryCondition,
    _apply_zero_ghosts,
)
from cosmic_foundry.theory.discrete.discrete_field import (
    DiscreteField,
    _CallableDiscreteField,
)
from cosmic_foundry.theory.discrete.discretization import Discretization
from cosmic_foundry.theory.discrete.numerical_flux import NumericalFlux


class FVMDiscretization(Discretization[sympy.Expr]):
    """Finite-volume discrete operator for a divergence-form equation on a
    CartesianMesh.

    Maps a DiscreteField of cell averages to a DiscreteField of average discrete
    divergences:

        (Lₕ U)(i) = (1/|Ωᵢ|) ∮_∂Ωᵢ F·n̂ dA
                  = (1/|Ωᵢ|) Σ_a [F(U)((a, i)) − F(U)((a, i−eₐ))]

    approximating (1/|Ωᵢ|) ∫_Ωᵢ L φ dV at convergence order p = numerical_flux.order.
    U is assumed to hold cell-average values; NumericalFlux stencils operate on
    averages directly.

    When a DiscreteBoundaryCondition is supplied, ghost cells are applied via
    bc.extend(U, mesh) before face fluxes are evaluated.  When no BC is supplied,
    zero-valued ghost cells are used.

    continuous_operator is auto-derived as ∇·(numerical_flux.continuous_operator).

    Parameters
    ----------
    numerical_flux:
        The NumericalFlux approximating the face-averaged flux F·n̂·|A|.
    boundary_condition:
        Optional DiscreteBoundaryCondition; when supplied, ghost cells are
        applied in __call__.
    """

    def __init__(
        self,
        numerical_flux: NumericalFlux[Any],
        boundary_condition: DiscreteBoundaryCondition | None = None,
    ) -> None:
        super().__init__(boundary_condition)
        self._numerical_flux = numerical_flux

    @property
    def order(self) -> int:
        return self._numerical_flux.order

    @property
    def continuous_operator(self) -> DifferentialOperator:
        return DivergenceComposition(self._numerical_flux.continuous_operator)

    def __call__(self, U: DiscreteField[sympy.Expr]) -> DiscreteField[sympy.Expr]:
        """Apply the discrete divergence operator; return cell residuals."""
        mesh = cast(CartesianMesh, U.mesh)
        if self._boundary_condition is not None:
            U = self._boundary_condition.extend(U, mesh)
        else:
            U = _apply_zero_ghosts(U, mesh)
        face_fluxes = self._numerical_flux(U)
        vol = mesh.cell_volume
        div = CartesianExteriorDerivative(mesh, degree=2)(face_fluxes)
        return _CallableDiscreteField(mesh, lambda idx: div(idx) / vol)


__all__ = ["FVMDiscretization"]
