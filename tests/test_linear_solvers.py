"""Linear-solver correctness claims.

The module owns residual correctness checks for concrete linear solvers on
small assembled finite-volume systems. Discrete-operator order and
manufactured-solution convergence claims live in tests/test_discrete_operators.py.
"""

from __future__ import annotations

import math
from typing import Any

import pytest
import sympy

from cosmic_foundry.computation import tensor
from cosmic_foundry.computation.backends import NumpyBackend
from cosmic_foundry.computation.decompositions.svd_factorization import SVDFactorization
from cosmic_foundry.computation.solvers.dense_cg_solver import DenseCGSolver
from cosmic_foundry.computation.solvers.dense_gauss_seidel_solver import (
    DenseGaussSeidelSolver,
)
from cosmic_foundry.computation.solvers.dense_gmres_solver import DenseGMRESSolver
from cosmic_foundry.computation.solvers.dense_jacobi_solver import DenseJacobiSolver
from cosmic_foundry.computation.solvers.dense_lu_solver import DenseLUSolver
from cosmic_foundry.computation.solvers.dense_svd_solver import DenseSVDSolver
from cosmic_foundry.computation.solvers.direct_solver import DirectSolver
from cosmic_foundry.computation.tensor import Tensor
from cosmic_foundry.geometry.cartesian_mesh import CartesianMesh
from cosmic_foundry.geometry.euclidean_manifold import EuclideanManifold
from cosmic_foundry.theory.discrete import (
    AdvectionDiffusionFlux,
    AdvectiveFlux,
    DiffusiveFlux,
    DirichletGhostCells,
    DivergenceFormDiscretization,
    PeriodicGhostCells,
)
from cosmic_foundry.theory.discrete.discrete_field import _CallableDiscreteField
from tests.claims import Claim

_DIMS = [1, 2, 3]
_SOLVER_MESH_N = {1: 8, 2: 4, 3: 3}
_NP_BACKEND = NumpyBackend()
_ASSEMBLER = DirectSolver(SVDFactorization())


class _DiscreteApplyOperator:
    def __init__(self, disc: Any, mesh: CartesianMesh) -> None:
        self._disc = disc
        self._mesh = mesh
        self._n = mesh.n_cells
        self._shape = mesh.shape

    def _to_flat(self, idx: tuple[int, ...]) -> int:
        flat, stride = 0, 1
        for axis, cell in enumerate(idx):
            flat += cell * stride
            stride *= self._shape[axis]
        return flat

    def _to_multi(self, flat: int) -> tuple[int, ...]:
        idx = []
        for size in self._shape:
            idx.append(flat % size)
            flat //= size
        return tuple(idx)

    def apply(self, u: Tensor) -> Tensor:
        field = _CallableDiscreteField(
            self._mesh, lambda idx: float(u[self._to_flat(idx)])
        )
        result = self._disc(field)
        values = [float(result(self._to_multi(i))) for i in range(self._n)]
        return Tensor(values, backend=u.backend)


class _DiscreteLinearOperator:
    """LinearOperator whose matrix is assembled through DirectSolver._assemble."""

    def __init__(self, disc: Any, mesh: CartesianMesh) -> None:
        self._n = mesh.n_cells
        self._apply_op = _DiscreteApplyOperator(disc, mesh)
        self._matrices: dict[int, Tensor] = {}

    def _matrix(self, backend: Any) -> Tensor:
        key = id(backend)
        if key not in self._matrices:
            self._matrices[key] = _ASSEMBLER._assemble(
                self._apply_op, Tensor.zeros(self._n, backend=backend)
            )
        return self._matrices[key]

    def apply(self, u: Tensor) -> Tensor:
        return self._matrix(u.backend) @ u

    def diagonal(self, backend: Any) -> Tensor:
        matrix = self._matrix(backend)
        return Tensor([float(matrix[i, i]) for i in range(self._n)], backend=backend)

    def row_abs_sums(self, backend: Any) -> Tensor:
        matrix = self._matrix(backend)
        return Tensor(
            [
                sum(abs(float(matrix[i, j])) for j in range(self._n))
                for i in range(self._n)
            ],
            backend=backend,
        )


class _SolverClaim(Claim[float]):
    """Claim: solver residual < tol after solve on the given Discretization.

    Verifies that ‖b − Au‖₂ < tol after solve returns.  disc is pre-built
    with its BC; assembled to a LinearOperator at check time.
    """

    def __init__(self, solver: Any, disc: Any, mesh: CartesianMesh) -> None:
        self._solver = solver
        self._disc = disc
        self._mesh = mesh

    @property
    def description(self) -> str:
        n = math.prod(self._mesh.shape)
        ndim = len(self._mesh.shape)
        return (
            f"{type(self._solver).__name__}/"
            f"{type(self._disc).__name__}(order={self._disc.order})/N={n}/{ndim}D"
        )

    def check(self, fma_rate: float) -> None:
        n = math.prod(self._mesh.shape)
        op = _DiscreteLinearOperator(self._disc, self._mesh)
        b = Tensor([1.0] * n, backend=_NP_BACKEND)
        u = self._solver.solve(op, b)
        residual = tensor.norm(b - op.apply(u)).get()
        tol = getattr(self._solver, "_tol", 1e-10)
        assert residual < tol, f"Did not converge: residual {residual:.3e}"


class _DirectSolverClaim(Claim[float]):
    """Claim: direct solver residual < tol after one factorization pass.

    disc is pre-built with its BC; assembled to a LinearOperator at check time.
    For PeriodicGhostCells the RHS is a zero-mean product of sines so the system
    is consistent (in the column space of the periodic advection operator).
    Works for any spatial dimensionality.
    """

    def __init__(self, solver: Any, disc: Any, mesh: CartesianMesh) -> None:
        self._solver = solver
        self._disc = disc
        self._mesh = mesh

    @property
    def description(self) -> str:
        n = math.prod(self._mesh.shape)
        ndim = len(self._mesh.shape)
        periodic = isinstance(self._disc.boundary_condition, PeriodicGhostCells)
        suffix = "/periodic" if periodic else ""
        return (
            f"{type(self._solver).__name__}/"
            f"{type(self._disc).__name__}(order={self._disc.order})"
            f"/N={n}/{ndim}D{suffix}"
        )

    def check(self, fma_rate: float) -> None:
        n = math.prod(self._mesh.shape)
        op = _DiscreteLinearOperator(self._disc, self._mesh)
        if isinstance(self._disc.boundary_condition, PeriodicGhostCells):
            shape = self._mesh.shape
            ndim = len(shape)

            def _idx(flat: int) -> tuple[int, ...]:
                out = []
                for s in shape:
                    out.append(flat % s)
                    flat //= s
                return tuple(out)

            # Use sum mode sin(2π·(x₁+…+xd)) as RHS.  Tensor-product modes
            # sin(2πx)·sin(2πy) contain Fourier components (k,-k) which are in
            # the null space of v·(∂/∂x+∂/∂y) (eigenvalue ∝ sin(2πk/N)+sin(-2πk/N)=0).
            # The sum mode has only the (1,1,…,1) Fourier component, whose eigenvalue
            # i·v/h·d·sin(2π/N) ≠ 0 for N ≥ 3, placing b safely in the column space.
            # The mean over a full period is exactly zero; no subtraction needed.
            raw = [
                math.sin(
                    2
                    * math.pi
                    * sum(float(self._mesh.coordinate(_idx(i))[k]) for k in range(ndim))
                )
                for i in range(n)
            ]
            b = Tensor(raw)
        else:
            b = Tensor([1.0] * n)
        u = self._solver.solve(op, b)
        residual = tensor.norm(b - op.apply(u)).get()
        assert residual < 1e-10, f"Direct solve residual {residual:.3e} >= 1e-10"


# ---------------------------------------------------------------------------
# Solver registries
# ---------------------------------------------------------------------------

# DiffusiveFlux → SPD matrix (DirichletBC): all solvers including CG.
# AdvectiveFlux → rank-(N-1) circulant (PeriodicBC): direct solvers only.
# AdvectionDiffusionFlux → non-symmetric (DirichletBC): no CG.
_SOLVERS = [
    DenseJacobiSolver(tol=1e-8),
    DenseGaussSeidelSolver(tol=1e-8),
    DenseGMRESSolver(tol=1e-8),
]
_SPD_SOLVERS = [DenseCGSolver(tol=1e-8)]
_DIRECT_SOLVERS = [DenseLUSolver(), DenseSVDSolver()]


# ---------------------------------------------------------------------------
# Claim generation: solver residual correctness
# ---------------------------------------------------------------------------

_CLAIMS: list[Claim[float]] = []

for _ndim in _DIMS:
    _manifold = EuclideanManifold(_ndim)
    _n_per_axis = _SOLVER_MESH_N[_ndim]
    _solver_mesh = CartesianMesh(
        origin=tuple(sympy.Rational(0) for _ in range(_ndim)),
        spacing=tuple(sympy.Rational(1, _n_per_axis) for _ in range(_ndim)),
        shape=(_n_per_axis,) * _ndim,
    )

    _diff_fluxes = [
        DiffusiveFlux(DiffusiveFlux.min_order, _manifold),
        DiffusiveFlux(DiffusiveFlux.min_order + DiffusiveFlux.order_step, _manifold),
    ]
    _adv_fluxes = [
        AdvectiveFlux(AdvectiveFlux.min_order, _manifold),
        AdvectiveFlux(AdvectiveFlux.min_order + AdvectiveFlux.order_step, _manifold),
    ]
    _adv_diff_fluxes = [
        AdvectionDiffusionFlux(AdvectionDiffusionFlux.min_order, _manifold),
        AdvectionDiffusionFlux(
            AdvectionDiffusionFlux.min_order + AdvectionDiffusionFlux.order_step,
            _manifold,
        ),
    ]

    for _f in _diff_fluxes:
        _disc = DivergenceFormDiscretization(_f, DirichletGhostCells())
        for _s in [*_SOLVERS, *_SPD_SOLVERS]:
            _CLAIMS.append(_SolverClaim(_s, _disc, _solver_mesh))
        for _s in _DIRECT_SOLVERS:
            _CLAIMS.append(_DirectSolverClaim(_s, _disc, _solver_mesh))

    for _f in _adv_fluxes:
        _disc = DivergenceFormDiscretization(_f, PeriodicGhostCells())
        for _s in _DIRECT_SOLVERS:
            _CLAIMS.append(_DirectSolverClaim(_s, _disc, _solver_mesh))

    for _f in _adv_diff_fluxes:
        _disc = DivergenceFormDiscretization(_f, DirichletGhostCells())
        for _s in _SOLVERS:
            _CLAIMS.append(_SolverClaim(_s, _disc, _solver_mesh))
        for _s in _DIRECT_SOLVERS:
            _CLAIMS.append(_DirectSolverClaim(_s, _disc, _solver_mesh))


@pytest.mark.parametrize("claim", _CLAIMS, ids=[c.description for c in _CLAIMS])
def test_correctness(claim: Claim[float], fma_rate: float) -> None:
    claim.check(fma_rate)
