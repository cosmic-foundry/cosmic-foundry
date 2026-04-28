"""FVMDiscretization: assemble a DiscreteOperator from a NumericalFlux."""

from __future__ import annotations

from typing import Any, cast

import sympy

from cosmic_foundry.geometry.cartesian_mesh import CartesianMesh
from cosmic_foundry.theory.continuous.differential_form import ZeroForm
from cosmic_foundry.theory.continuous.differential_operator import DifferentialOperator
from cosmic_foundry.theory.discrete.discrete_boundary_condition import (
    DiscreteBoundaryCondition,
)
from cosmic_foundry.theory.discrete.discrete_field import (
    DiscreteField,
    _CallableDiscreteField,
)
from cosmic_foundry.theory.discrete.discrete_operator import DiscreteOperator
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
        return ZeroForm(phi.manifold, div, phi.symbols)


def _flat_to_multi(flat: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    idx = []
    for n in shape:
        idx.append(flat % n)
        flat //= n
    return tuple(idx)


class _AssembledFVMOperator(DiscreteOperator[sympy.Expr]):
    """Assembled discrete divergence operator produced by FVMDiscretization.__call__.

    Maps a DiscreteField of cell averages to a DiscreteField of average discrete
    divergences:

        (Lₕ U)(i) = (1/|Ωᵢ|) ∮_∂Ωᵢ F·n̂ dA
                  = (1/|Ωᵢ|) Σ_a [F(U)((a, i)) − F(U)((a, i−eₐ))]

    approximating (1/|Ωᵢ|) ∫_Ωᵢ L φ dV at convergence order p = numerical_flux.order.
    U is assumed to hold cell-average values; NumericalFlux stencils operate on
    averages directly.

    When a DiscreteBoundaryCondition is supplied (via FVMDiscretization), ghost
    cells are applied via bc.extend(U, mesh) before face fluxes are evaluated,
    making the operator well-defined for all cells including those adjacent to
    the boundary.  When no BC is supplied, zero-valued ghost cells are used.

    continuous_operator is auto-derived as ∇·(numerical_flux.continuous_operator).

    stiffness_values, row_indices, col_indices are precomputed at construction time
    by applying the operator symbolically to a field of sympy symbols and reading
    off coefficients.  Operator.assemble() uses these for a single scatter operation
    instead of the N×N basis-vector loop.
    """

    def __init__(
        self,
        numerical_flux: NumericalFlux[sympy.Expr],
        mesh: CartesianMesh,
        bc: DiscreteBoundaryCondition | None = None,
    ) -> None:
        self._numerical_flux = numerical_flux
        self._bc = bc
        self._stiffness_values, self._row_indices, self._col_indices = (
            self._build_stiffness(mesh)
        )

    def _build_stiffness(
        self, mesh: CartesianMesh
    ) -> tuple[list[float], list[int], list[int]]:
        """Apply operator symbolically; extract nonzero stiffness coefficients."""
        n = mesh.n_cells
        shape = mesh.shape
        u_syms = [sympy.Symbol(f"_u{j}") for j in range(n)]

        def _to_flat(idx: tuple[int, ...]) -> int:
            flat, stride = 0, 1
            for a, i in enumerate(idx):
                flat += i * stride
                stride *= shape[a]
            return flat

        sym_field: _CallableDiscreteField[sympy.Expr] = _CallableDiscreteField(
            mesh, lambda idx: u_syms[_to_flat(idx)]
        )
        result = self(sym_field)

        vals: list[float] = []
        rows: list[int] = []
        cols: list[int] = []
        for i in range(n):
            expr = result(_flat_to_multi(i, shape))  # type: ignore[arg-type]
            for j, sym in enumerate(u_syms):
                c = float(expr.coeff(sym))
                if c != 0.0:
                    vals.append(c)
                    rows.append(i)
                    cols.append(j)
        return vals, rows, cols

    @property
    def stiffness_values(self) -> list[float]:
        return self._stiffness_values

    @property
    def row_indices(self) -> list[int]:
        return self._row_indices

    @property
    def col_indices(self) -> list[int]:
        return self._col_indices

    @property
    def order(self) -> int:
        return self._numerical_flux.order

    @property
    def continuous_operator(self) -> DifferentialOperator:
        return _DivergenceComposition(self._numerical_flux.continuous_operator)

    def __call__(self, U: DiscreteField[sympy.Expr]) -> DiscreteField[sympy.Expr]:
        """Apply the assembled operator; return cell residuals as DiscreteField."""
        mesh = cast(CartesianMesh, U.mesh)
        if self._bc is not None:
            U = self._bc.extend(U, mesh)
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

    def __call__(self) -> _AssembledFVMOperator:
        """Produce the assembled discrete operator."""
        return _AssembledFVMOperator(
            self._numerical_flux,
            self._mesh,  # type: ignore[arg-type]
            self._boundary_condition,
        )


__all__ = ["FVMDiscretization"]
