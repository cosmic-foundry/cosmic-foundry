"""Linear-solver correctness claims.

The module owns residual correctness checks for concrete linear solvers on
small assembled finite-volume systems. Discrete-operator order and
manufactured-solution convergence claims live in tests/test_discrete_operators.py.
"""

from __future__ import annotations

import math
import time
from typing import Any

import pytest
import sympy

from cosmic_foundry.computation.algorithm_capabilities import (
    DecompositionField,
    LinearOperatorEvidence,
    SolveRelationField,
    decomposition_descriptor_from_linear_operator_descriptor,
    linear_operator_descriptor_from_assembled_operator,
)
from cosmic_foundry.computation.backends import NumpyBackend
from cosmic_foundry.computation.decompositions import (
    select_decomposition_for_descriptor,
)
from cosmic_foundry.computation.decompositions.lu_factorization import LUFactorization
from cosmic_foundry.computation.decompositions.svd_factorization import SVDFactorization
from cosmic_foundry.computation.solvers.capabilities import (
    select_least_squares_solver_for_descriptor,
)
from cosmic_foundry.computation.solvers.dense_cg_solver import DenseCGSolver
from cosmic_foundry.computation.solvers.dense_gauss_seidel_solver import (
    DenseGaussSeidelSolver,
)
from cosmic_foundry.computation.solvers.dense_gmres_solver import DenseGMRESSolver
from cosmic_foundry.computation.solvers.dense_jacobi_solver import DenseJacobiSolver
from cosmic_foundry.computation.solvers.dense_lu_solver import DenseLUSolver
from cosmic_foundry.computation.solvers.dense_svd_solver import DenseSVDSolver
from cosmic_foundry.computation.solvers.direct_solver import DirectSolver
from cosmic_foundry.computation.solvers.least_squares_solver import (
    DenseSVDLeastSquaresSolver,
)
from cosmic_foundry.computation.solvers.relations import LeastSquaresRelation
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
from tests.claims import BatchedFailure, Claim, ExecutionPlan

_DIMS = [1, 2, 3]
_SOLVER_MESH_N = {1: 8, 2: 4, 3: 3}
_NP_BACKEND = NumpyBackend()
_ASSEMBLER = DirectSolver(SVDFactorization())
_PERF_OVERHEAD = 200.0


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


class _RectangularLinearOperator:
    def __init__(self, matrix: tuple[tuple[float, ...], ...]) -> None:
        self._matrix = matrix
        self._rows = len(matrix)
        self._columns = len(matrix[0])

    def apply(self, u: Tensor) -> Tensor:
        return Tensor(
            [
                sum(
                    self._matrix[row][column] * float(u[column])
                    for column in range(self._columns)
                )
                for row in range(self._rows)
            ],
            backend=u.backend,
        )


def _flux_name(disc: Any) -> str:
    return type(disc._numerical_flux).__name__


def _batched_rhs_values(
    mesh: CartesianMesh,
    lane_indices: tuple[int, ...],
    *,
    source_batch: int,
    periodic: bool,
) -> list[list[float]]:
    n = mesh.n_cells
    if not periodic:
        return [
            [1.0 + 0.05 * math.sin((col + 1) * (row + 1)) for col in lane_indices]
            for row in range(n)
        ]

    shape = mesh.shape
    ndim = len(shape)

    def _idx(flat: int) -> tuple[int, ...]:
        out = []
        for s in shape:
            out.append(flat % s)
            flat //= s
        return tuple(out)

    base = [
        math.sin(
            2 * math.pi * sum(float(mesh.coordinate(_idx(row))[k]) for k in range(ndim))
        )
        for row in range(n)
    ]
    return [
        [
            (1.0 + 0.1 * col / max(source_batch - 1, 1)) * base[row]
            for col in lane_indices
        ]
        for row in range(n)
    ]


def _column(matrix: list[list[float]], col: int, backend: Any) -> Tensor:
    return Tensor([row[col] for row in matrix], backend=backend)


def _matrix_from_columns(columns: list[Tensor], backend: Any) -> Tensor:
    return Tensor(
        [
            [float(columns[col][row]) for col in range(len(columns))]
            for row in range(columns[0].shape[0])
        ],
        backend=backend,
    )


def _residual_column_errors(residual: Tensor) -> list[float]:
    n, batch_size = residual.shape
    return [
        math.sqrt(sum(float(residual[row, col]) ** 2 for row in range(n)))
        for col in range(batch_size)
    ]


class _SolverClaim(Claim[ExecutionPlan]):
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

    def check(self, execution_plan: ExecutionPlan) -> None:
        n = math.prod(self._mesh.shape)
        op = _DiscreteLinearOperator(self._disc, self._mesh)
        source_batch = execution_plan.batch_size_for(
            fmas_per_case=n * n * 16.0,
            min_batch=2,
            max_batch=4,
        )
        lane_indices = execution_plan.batch_indices_for(
            source_batch, label=self.description
        )
        b_values = _batched_rhs_values(
            self._mesh,
            lane_indices,
            source_batch=source_batch,
            periodic=False,
        )
        b = Tensor(b_values, backend=_NP_BACKEND)
        solutions = [
            self._solver.solve(op, _column(b_values, col, _NP_BACKEND))
            for col in range(len(lane_indices))
        ]
        u = _matrix_from_columns(solutions, _NP_BACKEND)
        residual = b - (op._matrix(_NP_BACKEND) @ u)
        errors = _residual_column_errors(residual)
        worst_col, worst_error = max(enumerate(errors), key=lambda item: item[1])
        tol = getattr(self._solver, "_tol", 1e-10)
        assert worst_error < tol, BatchedFailure(
            claim=self.description,
            device_kind=execution_plan.device_kind,
            batch_size=source_batch,
            batch_index=lane_indices[worst_col],
            method=type(self._solver).__name__,
            order=self._disc.order,
            problem=_flux_name(self._disc),
            parameters={"n": n, "ndim": len(self._mesh.shape)},
            actual=worst_error,
            expected=0.0,
            error=worst_error,
            tolerance=tol,
        ).format()


class _SVDLeastSquaresClaim(Claim[ExecutionPlan]):
    """Claim: SVD factorization solves rectangular least-squares systems."""

    @property
    def description(self) -> str:
        return "SVDFactorization/rectangular_least_squares"

    def check(self, execution_plan: ExecutionPlan) -> None:
        _assert_rectangular_least_squares_solution(
            self.description,
            "SVDFactorization",
            execution_plan,
            lambda a, b: SVDFactorization().factorize(a).solve(b),
        )


class _DenseSVDLeastSquaresSolverClaim(Claim[ExecutionPlan]):
    """Claim: dense SVD least-squares solver owns the rectangular interface."""

    @property
    def description(self) -> str:
        return "DenseSVDLeastSquaresSolver/rectangular_least_squares"

    def check(self, execution_plan: ExecutionPlan) -> None:
        solver = DenseSVDLeastSquaresSolver()
        _assert_rectangular_least_squares_solution(
            self.description,
            type(solver).__name__,
            execution_plan,
            lambda a, b: solver.solve(_least_squares_relation(a, b)),
        )


def _assert_rectangular_least_squares_solution(
    claim: str,
    method: str,
    execution_plan: ExecutionPlan,
    solve: Any,
) -> None:
    a = Tensor(
        (
            (1.0, 0.0),
            (0.0, 1.0),
            (1.0, 1.0),
        ),
        backend=_NP_BACKEND,
    )
    b = Tensor((1.0, 2.0, 4.0), backend=_NP_BACKEND)
    relation = _least_squares_relation(a, b)
    descriptor = relation.solve_relation_descriptor()
    assert descriptor.coordinate(SolveRelationField.DIM_X).value == 2
    assert descriptor.coordinate(SolveRelationField.DIM_Y).value == 3
    assert (
        descriptor.coordinate(SolveRelationField.OBJECTIVE_RELATION).value
        == "least_squares"
    )
    assert (
        descriptor.coordinate(SolveRelationField.ACCEPTANCE_RELATION).value
        == "objective_minimum"
    )
    if method != "SVDFactorization":
        assert (
            select_least_squares_solver_for_descriptor(descriptor)
            is DenseSVDLeastSquaresSolver
        )
    x = solve(a, b)
    expected = (4.0 / 3.0, 7.0 / 3.0)
    error = math.sqrt(sum((float(x[i]) - expected[i]) ** 2 for i in range(2)))
    residual = a @ x - b
    normal_residual = _transpose_matvec(a, residual)
    normal_error = math.sqrt(sum(float(normal_residual[i]) ** 2 for i in range(2)))
    tolerance = 1.0e-10
    assert error < tolerance and normal_error < tolerance, BatchedFailure(
        claim=claim,
        device_kind=execution_plan.device_kind,
        batch_size=1,
        batch_index=0,
        method=method,
        order=0,
        problem="least_squares",
        parameters={"rows": 3, "columns": 2},
        actual=max(error, normal_error),
        expected=0.0,
        error=max(error, normal_error),
        tolerance=tolerance,
    ).format()


def _least_squares_relation(a: Tensor, b: Tensor) -> LeastSquaresRelation:
    matrix = tuple(
        tuple(float(a[row, column]) for column in range(a.shape[1]))
        for row in range(a.shape[0])
    )
    return LeastSquaresRelation(
        LinearOperatorEvidence(_RectangularLinearOperator(matrix), b, matrix)
    )


def _transpose_matvec(matrix: Tensor, vector: Tensor) -> Tensor:
    rows, columns = matrix.shape
    return Tensor(
        [
            sum(float(matrix[row, column]) * float(vector[row]) for row in range(rows))
            for column in range(columns)
        ],
        backend=vector.backend,
    )


class _DecompositionDescriptorSelectionClaim(Claim[ExecutionPlan]):
    """Claim: decomposition descriptors select the factorization used to solve."""

    @property
    def description(self) -> str:
        return "Decomposition/descriptor_selection_solve"

    def check(self, execution_plan: ExecutionPlan) -> None:
        backend = execution_plan.backend
        full_rank_matrix = ((3.0, 1.0), (1.0, 2.0))
        full_rank_rhs = Tensor((1.0, 4.0), backend=backend)
        full_rank_descriptor = decomposition_descriptor_from_linear_operator_descriptor(
            linear_operator_descriptor_from_assembled_operator(
                _SmallMatrixOperator(full_rank_matrix),
                full_rank_rhs,
            )
        )
        assert (
            select_decomposition_for_descriptor(full_rank_descriptor) is LUFactorization
        )
        full_rank_solution = (
            LUFactorization()
            .factorize(Tensor(full_rank_matrix, backend=backend))
            .solve(full_rank_rhs)
        )
        full_rank_residual = (
            Tensor(full_rank_matrix, backend=backend) @ (full_rank_solution)
            - full_rank_rhs
        )
        assert _norm(full_rank_residual) < 1.0e-10

        singular_matrix = ((1.0, 1.0), (1.0, 1.0))
        singular_rhs = Tensor((2.0, 2.0), backend=backend)
        singular_descriptor = decomposition_descriptor_from_linear_operator_descriptor(
            linear_operator_descriptor_from_assembled_operator(
                _SmallMatrixOperator(singular_matrix),
                singular_rhs,
            )
        )
        assert (
            singular_descriptor.coordinate(
                DecompositionField.MATRIX_NULLITY_ESTIMATE
            ).value
            == 1
        )
        assert (
            select_decomposition_for_descriptor(singular_descriptor) is SVDFactorization
        )
        singular_solution = (
            SVDFactorization()
            .factorize(Tensor(singular_matrix, backend=backend))
            .solve(singular_rhs)
        )
        singular_residual = (
            Tensor(singular_matrix, backend=backend) @ (singular_solution)
            - singular_rhs
        )
        assert _norm(singular_residual) < 1.0e-10
        assert abs(float(singular_solution[0]) - float(singular_solution[1])) < 1.0e-10


class _SmallMatrixOperator:
    def __init__(self, matrix: tuple[tuple[float, ...], ...]) -> None:
        self._matrix = matrix

    def apply(self, u: Tensor) -> Tensor:
        return Tensor(
            [
                sum(row[column] * float(u[column]) for column in range(len(row)))
                for row in self._matrix
            ],
            backend=u.backend,
        )


def _norm(vector: Tensor) -> float:
    return math.sqrt(sum(float(vector[i]) ** 2 for i in range(vector.shape[0])))


class _DirectSolverClaim(Claim[ExecutionPlan]):
    """Claim: direct solver residual < tol across a batched RHS set.

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

    def check(self, execution_plan: ExecutionPlan) -> None:
        n = math.prod(self._mesh.shape)
        op = _DiscreteLinearOperator(self._disc, self._mesh)
        source_batch = execution_plan.batch_size_for(
            fmas_per_case=n * n * n * 4.0,
            min_batch=2,
            max_batch=4,
        )
        lane_indices = execution_plan.batch_indices_for(
            source_batch, label=self.description
        )
        periodic = isinstance(self._disc.boundary_condition, PeriodicGhostCells)
        b_values = _batched_rhs_values(
            self._mesh,
            lane_indices,
            source_batch=source_batch,
            periodic=periodic,
        )
        b = Tensor(b_values, backend=_NP_BACKEND)
        solutions = [
            self._solver.solve(op, _column(b_values, col, _NP_BACKEND))
            for col in range(len(lane_indices))
        ]
        u = _matrix_from_columns(solutions, _NP_BACKEND)
        residual = b - (op._matrix(_NP_BACKEND) @ u)
        errors = _residual_column_errors(residual)
        worst_col, worst_error = max(enumerate(errors), key=lambda item: item[1])
        assert worst_error < 1e-10, BatchedFailure(
            claim=self.description,
            device_kind=execution_plan.device_kind,
            batch_size=source_batch,
            batch_index=lane_indices[worst_col],
            method=type(self._solver).__name__,
            order=self._disc.order,
            problem=_flux_name(self._disc),
            parameters={
                "n": n,
                "ndim": len(self._mesh.shape),
                "periodic": periodic,
            },
            actual=worst_error,
            expected=0.0,
            error=worst_error,
            tolerance=1e-10,
        ).format()


class _CachedLUPerformanceClaim(Claim[ExecutionPlan]):
    """Claim: cached LU solves stay within the execution-plan Tensor roofline."""

    @property
    def description(self) -> str:
        return "DenseLUSolver/cached_factorization_solve"

    def check(self, execution_plan: ExecutionPlan) -> None:
        n = execution_plan.problem_size_for(
            work_fmas=lambda size: size**3 + 8.0 * size**2,
            min_size=8,
            max_size=32,
            label=self.description,
        )
        source_rhs_count = execution_plan.batch_size_for(
            fmas_per_case=2.0 * n**2,
            min_batch=2,
            max_batch=16,
        )
        lane_indices = execution_plan.batch_indices_for(
            source_rhs_count, label=self.description
        )
        rhs_count = len(lane_indices)
        backend = execution_plan.backend
        matrix = Tensor(
            [
                [2.0 if i == j else -1.0 if abs(i - j) == 1 else 0.0 for j in range(n)]
                for i in range(n)
            ],
            backend=backend,
        )
        rhs_values = [
            [1.0 + 0.05 * math.sin((col + 1) * (row + 1)) for col in lane_indices]
            for row in range(n)
        ]
        factorization = LUFactorization().factorize(matrix)

        def solve_all() -> Tensor:
            solutions = [
                factorization.solve(_column(rhs_values, col, backend))
                for col in range(rhs_count)
            ]
            result = _matrix_from_columns(solutions, backend)
            result.sync()
            return result

        best = float("inf")
        best_solution: Tensor | None = None
        for _ in range(5):
            t0 = time.perf_counter()
            solution = solve_all()
            elapsed = time.perf_counter() - t0
            if elapsed < best:
                best = elapsed
                best_solution = solution
        assert best_solution is not None
        residual = Tensor(rhs_values, backend=backend) - (matrix @ best_solution)
        errors = _residual_column_errors(residual)
        worst_col, worst_error = max(enumerate(errors), key=lambda item: item[1])
        assert worst_error < 1e-10, BatchedFailure(
            claim=self.description,
            device_kind=execution_plan.device_kind,
            batch_size=source_rhs_count,
            batch_index=lane_indices[worst_col],
            method="LUFactorization.solve",
            order=None,
            problem="tridiagonal_spd",
            parameters={"n": n},
            actual=worst_error,
            expected=0.0,
            error=worst_error,
            tolerance=1e-10,
        ).format()
        fmas = rhs_count * 2.0 * n**2
        roofline = fmas / execution_plan.fma_rate
        assert best <= _PERF_OVERHEAD * roofline, (
            f"{self.description}/{execution_plan.device_kind}: n={n}, "
            f"rhs_count={rhs_count}/{source_rhs_count}, {best:.3e}s actual, "
            f"{roofline:.3e}s Tensor roofline, "
            f"ratio={best / roofline:.1f}x > {_PERF_OVERHEAD:.1f}x"
        )


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

_CORRECTNESS_CLAIMS: list[Claim[ExecutionPlan]] = []
_CORRECTNESS_CLAIMS.append(_SVDLeastSquaresClaim())
_CORRECTNESS_CLAIMS.append(_DenseSVDLeastSquaresSolverClaim())
_CORRECTNESS_CLAIMS.append(_DecompositionDescriptorSelectionClaim())

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
            _CORRECTNESS_CLAIMS.append(_SolverClaim(_s, _disc, _solver_mesh))
        for _s in _DIRECT_SOLVERS:
            _CORRECTNESS_CLAIMS.append(_DirectSolverClaim(_s, _disc, _solver_mesh))

    for _f in _adv_fluxes:
        _disc = DivergenceFormDiscretization(_f, PeriodicGhostCells())
        for _s in _DIRECT_SOLVERS:
            _CORRECTNESS_CLAIMS.append(_DirectSolverClaim(_s, _disc, _solver_mesh))

    for _f in _adv_diff_fluxes:
        _disc = DivergenceFormDiscretization(_f, DirichletGhostCells())
        for _s in _SOLVERS:
            _CORRECTNESS_CLAIMS.append(_SolverClaim(_s, _disc, _solver_mesh))
        for _s in _DIRECT_SOLVERS:
            _CORRECTNESS_CLAIMS.append(_DirectSolverClaim(_s, _disc, _solver_mesh))

_PERFORMANCE_CLAIMS: list[Claim[ExecutionPlan]] = [_CachedLUPerformanceClaim()]


@pytest.mark.parametrize(
    "claim", _CORRECTNESS_CLAIMS, ids=[c.description for c in _CORRECTNESS_CLAIMS]
)
def test_correctness(
    claim: Claim[ExecutionPlan], execution_plan: ExecutionPlan
) -> None:
    claim.check(execution_plan)


@pytest.mark.parametrize(
    "claim", _PERFORMANCE_CLAIMS, ids=[c.description for c in _PERFORMANCE_CLAIMS]
)
def test_performance(
    claim: Claim[ExecutionPlan], execution_plan: ExecutionPlan
) -> None:
    claim.check(execution_plan)
