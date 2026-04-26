"""FVMDiscretization: assemble a DiscreteOperator from a NumericalFlux."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import sympy

from cosmic_foundry.computation.backends.python_backend import PythonBackend
from cosmic_foundry.computation.tensor import Tensor
from cosmic_foundry.geometry.cartesian_mesh import CartesianMesh
from cosmic_foundry.physics.state import State
from cosmic_foundry.theory.continuous.boundary_condition import BoundaryCondition
from cosmic_foundry.theory.continuous.differential_form import ZeroForm
from cosmic_foundry.theory.continuous.differential_operator import DifferentialOperator
from cosmic_foundry.theory.continuous.periodic_bc import PeriodicBC
from cosmic_foundry.theory.discrete.cell_field import CellField
from cosmic_foundry.theory.discrete.discrete_field import DiscreteField
from cosmic_foundry.theory.discrete.discrete_operator import DiscreteOperator
from cosmic_foundry.theory.discrete.discretization import Discretization
from cosmic_foundry.theory.discrete.mesh import Mesh
from cosmic_foundry.theory.discrete.numerical_flux import NumericalFlux


class _GhostedField(CellField[sympy.Expr]):
    """Cell field extended beyond mesh bounds by a ghost-cell rule."""

    def __init__(
        self,
        mesh: CartesianMesh,
        fn: Callable[[tuple[int, ...]], sympy.Expr],
    ) -> None:
        self._mesh = mesh
        self._fn = fn

    @property
    def mesh(self) -> CartesianMesh:
        return self._mesh

    def __call__(self, idx: tuple[int, ...]) -> sympy.Expr:  # type: ignore[override]
        return self._fn(idx)


def _apply_dirichlet_ghosts(
    U: DiscreteField[sympy.Expr],
    mesh: CartesianMesh,
) -> _GhostedField:
    """Extend U with homogeneous Dirichlet ghost cells via odd reflection.

    For each axis a and mesh size N = shape[a]:
        U(i < 0)  → −U(−1 − i)   (left ghost: odd reflection about face at 0)
        U(i >= N) → −U(2N − 1 − i) (right ghost: odd reflection about face at N)
    Corners (out of bounds in multiple axes) are handled recursively.
    """
    shape = mesh._shape

    def extended(idx: tuple[int, ...]) -> sympy.Expr:
        for a, (i, N) in enumerate(zip(idx, shape, strict=True)):
            if i < 0:
                reflected = idx[:a] + (-1 - i,) + idx[a + 1 :]
                return -extended(reflected)
            if i >= N:
                reflected = idx[:a] + (2 * N - 1 - i,) + idx[a + 1 :]
                return -extended(reflected)
        return U(idx)  # type: ignore[arg-type]

    return _GhostedField(mesh, extended)


def _apply_zero_ghosts(
    U: DiscreteField[sympy.Expr],
    mesh: CartesianMesh,
) -> _GhostedField:
    """Extend U with zero-valued ghost cells for no-BC operator evaluation.

    Semantically identical to evaluating a field that returns 0 for all
    out-of-bounds indices — which is what unit basis vectors in
    Discretization.assemble already do naturally.
    """
    shape = mesh._shape

    def extended(idx: tuple[int, ...]) -> sympy.Expr:
        for _a, (i, N) in enumerate(zip(idx, shape, strict=True)):
            if i < 0 or i >= N:
                return sympy.Integer(0)
        return U(idx)  # type: ignore[arg-type]

    return _GhostedField(mesh, extended)


def _apply_periodic_ghosts(
    U: DiscreteField[sympy.Expr],
    mesh: CartesianMesh,
) -> _GhostedField:
    """Extend U with periodic ghost cells via wrap-around.

    For each axis a and mesh size N = shape[a]:
        U(i < 0)  → U(N + i)   (left ghost: wrap to right end)
        U(i >= N) → U(i - N)   (right ghost: wrap to left end)
    """
    shape = mesh._shape

    def extended(idx: tuple[int, ...]) -> sympy.Expr:
        wrapped = tuple(i % N for i, N in zip(idx, shape, strict=True))
        return U(wrapped)  # type: ignore[arg-type]

    return _GhostedField(mesh, extended)


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


class _AssembledFVMOperator(DiscreteOperator[sympy.Expr]):
    """Assembled discrete divergence operator produced by FVMDiscretization.__call__.

    Maps cell-average DiscreteFields to a State holding discrete divergence values:

        (Lₕ U)(i) = (1/|Ωᵢ|) · ∮_∂Ωᵢ F·n̂ dA
                  = (1/|Ωᵢ|) · Σ_a [F(U)((a, i)) − F(U)((a, i−eₐ))]

    approximating Lφ = ∇·F(φ) at convergence order p = numerical_flux.order.
    The mesh is read from U.mesh at call time, making this operator applicable
    to symbolic meshes (for convergence testing) and concrete meshes alike.

    If a DirichletBC is supplied (via FVMDiscretization), homogeneous ghost
    cells are applied before computing face fluxes, making the operator
    well-defined for all cells including those adjacent to the boundary.

    continuous_operator is auto-derived as ∇·(numerical_flux.continuous_operator).
    """

    def __init__(
        self,
        numerical_flux: NumericalFlux[sympy.Expr],
        bc: BoundaryCondition | None = None,
    ) -> None:
        self._numerical_flux = numerical_flux
        self._bc = bc

    @property
    def order(self) -> int:
        return self._numerical_flux.order

    @property
    def continuous_operator(self) -> DifferentialOperator:
        return _DivergenceComposition(self._numerical_flux.continuous_operator)

    def __call__(self, U: DiscreteField[sympy.Expr]) -> State:
        """Apply the assembled operator; returns an eager cell-residual State."""
        mesh = cast(CartesianMesh, U.mesh)
        if isinstance(self._bc, PeriodicBC):
            U = _apply_periodic_ghosts(U, mesh)
        elif self._bc is not None:
            U = _apply_dirichlet_ghosts(U, mesh)
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

        residuals = []
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
            residuals.append(total / mesh.cell_volume)

        return State(mesh, Tensor(residuals, backend=PythonBackend()))


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
        Optional DirichletBC; when supplied, ghost cells are applied in
        __call__ and assemble (inherited from Discretization) uses the full
        boundary-aware operator.
    """

    def __init__(
        self,
        mesh: Mesh,
        numerical_flux: NumericalFlux[Any],
        boundary_condition: BoundaryCondition | None = None,
    ) -> None:
        super().__init__(mesh, boundary_condition)
        self._numerical_flux = numerical_flux

    def __call__(self) -> _AssembledFVMOperator:
        """Produce the assembled discrete operator."""
        return _AssembledFVMOperator(self._numerical_flux, self._boundary_condition)


__all__ = ["FVMDiscretization"]
