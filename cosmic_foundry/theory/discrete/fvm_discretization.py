"""FVMDiscretization: finite-volume DiscreteOperator on a CartesianMesh."""

from __future__ import annotations

from typing import Any, cast

import sympy

from cosmic_foundry.geometry.cartesian_mesh import CartesianMesh
from cosmic_foundry.theory.continuous.differential_operator import (
    DifferentialOperator,
    DivergenceComposition,
)
from cosmic_foundry.theory.discrete.discrete_boundary_condition import (
    DiscreteBoundaryCondition,
)
from cosmic_foundry.theory.discrete.discrete_field import (
    DiscreteField,
    _CallableDiscreteField,
)
from cosmic_foundry.theory.discrete.discretization import Discretization
from cosmic_foundry.theory.discrete.mesh import Mesh
from cosmic_foundry.theory.discrete.numerical_flux import NumericalFlux


def _apply_zero_ghosts(
    U: DiscreteField[sympy.Expr],
    mesh: CartesianMesh,
) -> DiscreteField[sympy.Expr]:
    """Extend U with zero-valued ghost cells for no-BC operator evaluation."""
    shape = mesh.shape

    def extended(idx: tuple[int, ...]) -> sympy.Expr:
        for i, N in zip(idx, shape, strict=True):
            if i < 0 or i >= N:
                return sympy.Integer(0)
        return U(idx)  # type: ignore[arg-type]

    return _CallableDiscreteField(mesh, extended)


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
    mesh:
        The CartesianMesh on which the scheme is defined.
    numerical_flux:
        The NumericalFlux approximating the face-averaged flux F·n̂·|A|.
    boundary_condition:
        Optional DiscreteBoundaryCondition; when supplied, ghost cells are
        applied in __call__.
    """

    def __init__(
        self,
        mesh: Mesh,
        numerical_flux: NumericalFlux[Any],
        boundary_condition: DiscreteBoundaryCondition | None = None,
    ) -> None:
        super().__init__(mesh, boundary_condition)
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
        ndim = len(mesh._shape)
        shape = mesh.shape

        def _to_multi(flat: int) -> tuple[int, ...]:
            idx = []
            for a in range(ndim):
                idx.append(flat % shape[a])
                flat //= shape[a]
            return tuple(idx)

        vol = mesh.cell_volume
        residuals: list[sympy.Expr] = []
        for flat_i in range(mesh.n_cells):
            idx = _to_multi(flat_i)
            total: sympy.Expr = sympy.Integer(0)
            for axis in range(ndim):
                idx_low: tuple[int, ...] = (
                    idx[:axis] + (idx[axis] - 1,) + idx[axis + 1 :]
                )
                total = (
                    total
                    + face_fluxes((axis, idx))  # type: ignore[arg-type]
                    - face_fluxes((axis, idx_low))  # type: ignore[arg-type]
                )
            residuals.append(total / vol)

        residuals_frozen = tuple(residuals)

        def lookup(idx: tuple[int, ...]) -> sympy.Expr:
            flat = 0
            stride = 1
            for a, i in enumerate(idx):
                flat += i * stride
                stride *= shape[a]
            return residuals_frozen[flat]

        return _CallableDiscreteField(mesh, lookup)


__all__ = ["FVMDiscretization"]
