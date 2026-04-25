"""FVMDiscretization: assemble a DiscreteOperator from a DivergenceFormEquation."""

from __future__ import annotations

from typing import Any, cast

import sympy

from cosmic_foundry.geometry.cartesian_mesh import CartesianMesh
from cosmic_foundry.theory.continuous.differential_operator import DifferentialOperator
from cosmic_foundry.theory.continuous.divergence_form_equation import (
    DivergenceFormEquation,
)
from cosmic_foundry.theory.discrete.discrete_operator import DiscreteOperator
from cosmic_foundry.theory.discrete.discretization import Discretization
from cosmic_foundry.theory.discrete.lazy_mesh_function import LazyMeshFunction
from cosmic_foundry.theory.discrete.mesh import Mesh
from cosmic_foundry.theory.discrete.mesh_function import MeshFunction
from cosmic_foundry.theory.discrete.numerical_flux import NumericalFlux


class _AssembledFVMOperator(DiscreteOperator[sympy.Expr]):
    """Assembled discrete divergence operator produced by FVMDiscretization.__call__.

    Maps cell-average MeshFunctions to discrete divergence MeshFunctions:

        (Lₕ U)(i) = (1/|Ωᵢ|) · ∮_∂Ωᵢ F·n̂ dA
                  = (1/|Ωᵢ|) · Σ_a [F(U)((a, i)) − F(U)((a, i−eₐ))]

    approximating Lφ = ∇·F(φ) at convergence order p = numerical_flux.order.
    The mesh is read from U.mesh at call time, making this operator applicable
    to symbolic meshes (for convergence testing) and concrete meshes alike.

    Carries continuous_operator = L (the DivergenceFormEquation passed to
    FVMDiscretization.__call__) so the commutation diagram is traceable.
    """

    def __init__(
        self,
        numerical_flux: NumericalFlux[sympy.Expr],
        continuous_operator: DifferentialOperator,
    ) -> None:
        self._numerical_flux = numerical_flux
        self._continuous_operator = continuous_operator

    @property
    def order(self) -> int:
        return self._numerical_flux.order

    @property
    def continuous_operator(self) -> DifferentialOperator:
        return self._continuous_operator

    def __call__(self, U: MeshFunction[sympy.Expr]) -> LazyMeshFunction[sympy.Expr]:
        """Apply the assembled operator; returns a lazy cell-residual MeshFunction."""
        mesh = cast(CartesianMesh, U.mesh)
        face_fluxes = self._numerical_flux(U)
        ndim = len(mesh._shape)

        def cell_residual(idx: Any) -> sympy.Expr:
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
            return total / mesh.cell_volume

        return LazyMeshFunction(mesh, cell_residual)


class FVMDiscretization(Discretization):
    """Finite-volume discretization of a DivergenceFormEquation on a CartesianMesh.

    FVMDiscretization(mesh, numerical_flux, boundary_condition) assembles the
    discrete operator Lₕ that makes the commutation diagram

        Lₕ Rₕ φ ≈ Rₕ L φ   (to O(hᵖ))

    hold for DifferentialEquation L and convergence order p = numerical_flux.order.
    Calling it with L produces an _AssembledFVMOperator carrying
    continuous_operator = L, closing the diagram from birth.

    Interior cells are evaluated correctly with no boundary input.  The
    boundary_condition is stored for later use; boundary face handling is
    deferred to a subsequent PR once a concrete LocalBoundaryCondition lands.

    Parameters
    ----------
    mesh:
        The CartesianMesh on which the scheme is defined.
    numerical_flux:
        The NumericalFlux approximating the face-averaged flux F·n̂·|A|.
    boundary_condition:
        The boundary condition on ∂Ω.  Stored but not yet applied.
    """

    def __init__(
        self,
        mesh: Mesh,
        numerical_flux: NumericalFlux[Any],
        boundary_condition: Any = None,
    ) -> None:
        self._mesh = mesh
        self._numerical_flux = numerical_flux
        self._boundary_condition = boundary_condition

    @property
    def mesh(self) -> Mesh:
        return self._mesh

    def __call__(self, L: DivergenceFormEquation) -> _AssembledFVMOperator:
        """Produce the assembled discrete operator for L."""
        return _AssembledFVMOperator(self._numerical_flux, L)


__all__ = ["FVMDiscretization"]
