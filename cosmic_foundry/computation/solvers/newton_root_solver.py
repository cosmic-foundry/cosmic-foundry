"""Newton solver for finite-dimensional root relations."""

from __future__ import annotations

from collections.abc import Callable

from cosmic_foundry.computation.algorithm_capabilities import (
    EvidenceSource,
    ParameterDescriptor,
    StructuredPredicate,
)
from cosmic_foundry.computation.decompositions.lu_factorization import LUFactorization
from cosmic_foundry.computation.solvers.coverage import (
    constrained_root_predicates,
    directional_derivative_root_predicates,
    fixed_point_root_predicates,
    unconstrained_root_predicates,
)
from cosmic_foundry.computation.solvers.dense_gmres_solver import DenseGMRESSolver
from cosmic_foundry.computation.solvers.relations import (
    FiniteDimensionalResidualRelation,
)
from cosmic_foundry.computation.tensor import Tensor, einsum, norm


class RootRelation(FiniteDimensionalResidualRelation):
    """Finite-dimensional residual relation ``F(x) = 0``."""

    jacobian: Callable[[Tensor], Tensor]

    def __init__(
        self,
        residual: Callable[[Tensor], Tensor],
        jacobian: Callable[[Tensor], Tensor],
        initial: Tensor,
        equality_constraint_gradients: Tensor | None = None,
    ) -> None:
        object.__setattr__(self, "residual", residual)
        object.__setattr__(self, "initial", initial)
        object.__setattr__(
            self,
            "equality_constraint_gradients",
            equality_constraint_gradients,
        )
        object.__setattr__(self, "jacobian", jacobian)

    def solve_relation_descriptor(
        self,
        *,
        map_linearity_defect: float | None = None,
        map_linearity_evidence: EvidenceSource = "unavailable",
        requested_residual_tolerance: float = 1.0e-8,
        requested_solution_tolerance: float = 1.0e-8,
        work_budget_fmas: float = 1.0e9,
        memory_budget_bytes: float = 1.0e9,
        device_kind: str = "cpu",
    ) -> ParameterDescriptor:
        """Project this root relation to primitive solve-relation coordinates."""
        return super().residual_relation_descriptor(
            target_is_zero=True,
            derivative_oracle_kind="jacobian_callback",
            map_linearity_defect=map_linearity_defect,
            map_linearity_evidence=map_linearity_evidence,
            matrix_representation_available=False,
            work_budget_fmas=work_budget_fmas,
            memory_budget_bytes=memory_budget_bytes,
            device_kind=device_kind,
            requested_residual_tolerance=requested_residual_tolerance,
            requested_solution_tolerance=requested_solution_tolerance,
        )


class DirectionalDerivativeRootRelation(FiniteDimensionalResidualRelation):
    """Finite-dimensional root relation with Jacobian-vector product evidence."""

    jvp: Callable[[Tensor, Tensor], Tensor]

    def __init__(
        self,
        residual: Callable[[Tensor], Tensor],
        jvp: Callable[[Tensor, Tensor], Tensor],
        initial: Tensor,
    ) -> None:
        object.__setattr__(self, "residual", residual)
        object.__setattr__(self, "initial", initial)
        object.__setattr__(self, "equality_constraint_gradients", None)
        object.__setattr__(self, "jvp", jvp)

    def solve_relation_descriptor(
        self,
        *,
        map_linearity_defect: float | None = None,
        map_linearity_evidence: EvidenceSource = "unavailable",
        requested_residual_tolerance: float = 1.0e-8,
        requested_solution_tolerance: float = 1.0e-8,
        work_budget_fmas: float = 1.0e9,
        memory_budget_bytes: float = 1.0e9,
        device_kind: str = "cpu",
    ) -> ParameterDescriptor:
        """Project this matrix-free root relation to primitive solve coordinates."""
        return super().residual_relation_descriptor(
            target_is_zero=True,
            derivative_oracle_kind="jvp",
            map_linearity_defect=map_linearity_defect,
            map_linearity_evidence=map_linearity_evidence,
            matrix_representation_available=False,
            work_budget_fmas=work_budget_fmas,
            memory_budget_bytes=memory_budget_bytes,
            device_kind=device_kind,
            requested_residual_tolerance=requested_residual_tolerance,
            requested_solution_tolerance=requested_solution_tolerance,
        )


class FixedPointRootRelation(FiniteDimensionalResidualRelation):
    """Finite-dimensional root relation with fixed-point iteration evidence."""

    fixed_point: Callable[[Tensor], Tensor]

    def __init__(
        self,
        residual: Callable[[Tensor], Tensor],
        fixed_point: Callable[[Tensor], Tensor],
        initial: Tensor,
    ) -> None:
        object.__setattr__(self, "residual", residual)
        object.__setattr__(self, "initial", initial)
        object.__setattr__(self, "equality_constraint_gradients", None)
        object.__setattr__(self, "fixed_point", fixed_point)

    def solve_relation_descriptor(
        self,
        *,
        map_linearity_defect: float | None = None,
        map_linearity_evidence: EvidenceSource = "unavailable",
        requested_residual_tolerance: float = 1.0e-8,
        requested_solution_tolerance: float = 1.0e-8,
        work_budget_fmas: float = 1.0e9,
        memory_budget_bytes: float = 1.0e9,
        device_kind: str = "cpu",
    ) -> ParameterDescriptor:
        """Project this fixed-point relation to primitive solve coordinates."""
        return super().residual_relation_descriptor(
            target_is_zero=True,
            derivative_oracle_kind="fixed_point_map",
            map_linearity_defect=map_linearity_defect,
            map_linearity_evidence=map_linearity_evidence,
            matrix_representation_available=False,
            work_budget_fmas=work_budget_fmas,
            memory_budget_bytes=memory_budget_bytes,
            device_kind=device_kind,
            requested_residual_tolerance=requested_residual_tolerance,
            requested_solution_tolerance=requested_solution_tolerance,
        )


class NewtonRootSolver:
    """Solve ``F(x) = 0`` by Newton iteration."""

    root_solver_coverage: tuple[tuple[StructuredPredicate, ...], ...] = (
        unconstrained_root_predicates() + constrained_root_predicates()
    )

    def __init__(
        self,
        *,
        max_iterations: int = 50,
        tolerance: float = 1e-12,
    ) -> None:
        self._max_iterations = max_iterations
        self._tolerance = tolerance
        self._lu = LUFactorization()

    def solve(
        self,
        relation: RootRelation,
    ) -> Tensor:
        """Return a root of the supplied residual relation."""
        x = relation.initial
        gram: Tensor | None = None
        if relation.equality_constraint_gradients is not None:
            gradients = relation.equality_constraint_gradients
            gram = einsum("ij,kj->ik", gradients, gradients)
        for _ in range(self._max_iterations):
            Fx = relation.residual(x)
            if self._small(Fx, x):
                break
            delta = self._lu.factorize(relation.jacobian(x)).solve(
                Tensor.zeros(x.shape[0], backend=x.backend) - Fx
            )
            if relation.equality_constraint_gradients is not None and gram is not None:
                gradients = relation.equality_constraint_gradients
                xi = self._lu.factorize(gram).solve(gradients @ delta)
                delta = delta - einsum("ij,i->j", gradients, xi)
            x = x + delta
            if self._small(delta, x):
                break
        return x

    def _small(self, vector: Tensor, scale: Tensor) -> bool:
        return float(norm(vector)) < self._tolerance * (1.0 + float(norm(scale)))


class _DirectionalDerivativeLinearization:
    def __init__(self, relation: DirectionalDerivativeRootRelation, point: Tensor):
        self._relation = relation
        self._point = point

    def apply(self, direction: Tensor) -> Tensor:
        return self._relation.jvp(self._point, direction)

    def diagonal(self, _backend: object) -> Tensor:
        raise NotImplementedError("JVP linearizations are matrix-free")

    def row_abs_sums(self, _backend: object) -> Tensor:
        raise NotImplementedError("JVP linearizations are matrix-free")


class MatrixFreeNewtonKrylovRootSolver:
    """Solve nonlinear roots by Newton iteration with JVP-only linear solves."""

    root_solver_coverage = directional_derivative_root_predicates()

    def __init__(
        self,
        *,
        max_iterations: int = 50,
        tolerance: float = 1e-12,
        krylov_tolerance: float = 1e-12,
        krylov_max_iterations: int = 50,
    ) -> None:
        self._max_iterations = max_iterations
        self._tolerance = tolerance
        self._krylov_tolerance = krylov_tolerance
        self._krylov_max_iterations = krylov_max_iterations

    def solve(self, relation: DirectionalDerivativeRootRelation) -> Tensor:
        """Return a root using only residual and Jacobian-vector products."""
        x = relation.initial
        for _ in range(self._max_iterations):
            Fx = relation.residual(x)
            if self._small(Fx, x):
                break
            linearization = _DirectionalDerivativeLinearization(relation, x)
            linear_solver = DenseGMRESSolver(
                tol=self._krylov_tolerance,
                max_iter=self._krylov_max_iterations,
                restart=x.shape[0],
            )
            delta = linear_solver.solve(
                linearization,
                Tensor.zeros(x.shape[0], backend=x.backend) - Fx,
            )
            x = x + delta
            if self._small(delta, x):
                break
        return x

    def _small(self, vector: Tensor, scale: Tensor) -> bool:
        return float(norm(vector)) < self._tolerance * (1.0 + float(norm(scale)))


class FixedPointRootSolver:
    """Solve roots by iterating a relation-provided fixed-point map."""

    root_solver_coverage = fixed_point_root_predicates()

    def __init__(
        self,
        *,
        max_iterations: int = 50,
        tolerance: float = 1e-12,
    ) -> None:
        self._max_iterations = max_iterations
        self._tolerance = tolerance

    def solve(self, relation: FixedPointRootRelation) -> Tensor:
        """Return a root using only a fixed-point map."""
        x = relation.initial
        for _ in range(self._max_iterations):
            x_next = relation.fixed_point(x)
            if self._small(x_next - x, x_next):
                x = x_next
                break
            x = x_next
            if self._small(relation.residual(x), x):
                break
        return x

    def _small(self, vector: Tensor, scale: Tensor) -> bool:
        return float(norm(vector)) < self._tolerance * (1.0 + float(norm(scale)))


__all__ = [
    "DirectionalDerivativeRootRelation",
    "FixedPointRootRelation",
    "FixedPointRootSolver",
    "MatrixFreeNewtonKrylovRootSolver",
    "NewtonRootSolver",
    "RootRelation",
]
