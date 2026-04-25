"""FVMDiscretization: assemble a DiscreteOperator from a NumericalFlux."""

from __future__ import annotations

from typing import Any, cast

import sympy

from cosmic_foundry.geometry.cartesian_mesh import CartesianMesh
from cosmic_foundry.theory.continuous.differential_form import ZeroForm
from cosmic_foundry.theory.continuous.differential_operator import DifferentialOperator
from cosmic_foundry.theory.discrete.discrete_operator import DiscreteOperator
from cosmic_foundry.theory.discrete.discretization import Discretization
from cosmic_foundry.theory.discrete.lazy_mesh_function import LazyMeshFunction
from cosmic_foundry.theory.discrete.mesh import Mesh
from cosmic_foundry.theory.discrete.mesh_function import MeshFunction
from cosmic_foundry.theory.discrete.numerical_flux import NumericalFlux


class _ConcreteZeroForm(ZeroForm[Any]):
    def __init__(
        self,
        manifold: Any,
        expr: sympy.Expr,
        symbols: tuple[sympy.Symbol, ...],
    ) -> None:
        self._manifold = manifold
        self._expr = expr
        self._symbols = symbols

    @property
    def manifold(self) -> Any:
        return self._manifold

    @property
    def expr(self) -> sympy.Expr:
        return self._expr

    @property
    def symbols(self) -> tuple[sympy.Symbol, ...]:
        return self._symbols


class _DivergenceComposition(DifferentialOperator[Any, ZeroForm[Any]]):
    """∇·F: the divergence of a DifferentialOperator F mapping ZeroForm → OneForm."""

    def __init__(self, flux_op: DifferentialOperator) -> None:
        self._flux_op = flux_op

    @property
    def manifold(self) -> Any:
        return self._flux_op.manifold

    @property
    def order(self) -> int:
        return self._flux_op.order + 1

    def __call__(self, phi: Any) -> ZeroForm[Any]:
        one_form = self._flux_op(phi)
        div: sympy.Expr = sum(
            (
                sympy.diff(one_form.component(i), one_form.symbols[i])
                for i in range(len(one_form.symbols))
            ),
            sympy.Integer(0),
        )
        return _ConcreteZeroForm(phi.manifold, div, phi.symbols)


class _AssembledFVMOperator(DiscreteOperator[sympy.Expr]):
    """Assembled discrete divergence operator produced by FVMDiscretization.__call__.

    Maps cell-average MeshFunctions to discrete divergence MeshFunctions:

        (Lₕ U)(i) = (1/|Ωᵢ|) · ∮_∂Ωᵢ F·n̂ dA
                  = (1/|Ωᵢ|) · Σ_a [F(U)((a, i)) − F(U)((a, i−eₐ))]

    approximating Lφ = ∇·F(φ) at convergence order p = numerical_flux.order.
    The mesh is read from U.mesh at call time, making this operator applicable
    to symbolic meshes (for convergence testing) and concrete meshes alike.

    continuous_operator is auto-derived as ∇·(numerical_flux.continuous_operator).
    """

    def __init__(self, numerical_flux: NumericalFlux[sympy.Expr]) -> None:
        self._numerical_flux = numerical_flux

    @property
    def order(self) -> int:
        return self._numerical_flux.order

    @property
    def continuous_operator(self) -> DifferentialOperator:
        return _DivergenceComposition(self._numerical_flux.continuous_operator)

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
    """Finite-volume discretization of a divergence-form equation on a CartesianMesh.

    FVMDiscretization(mesh, numerical_flux) assembles the discrete operator Lₕ
    that makes the commutation diagram

        Lₕ Rₕ φ ≈ Rₕ L φ   (to O(hᵖ))

    hold at convergence order p = numerical_flux.order.  Calling it produces an
    _AssembledFVMOperator whose continuous_operator is auto-derived as
    ∇·(numerical_flux.continuous_operator).

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

    def __call__(self) -> _AssembledFVMOperator:
        """Produce the assembled discrete operator."""
        return _AssembledFVMOperator(self._numerical_flux)


__all__ = ["FVMDiscretization"]
