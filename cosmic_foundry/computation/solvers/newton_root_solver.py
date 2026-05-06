"""Newton solver for finite-dimensional root relations."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from cosmic_foundry.computation.algorithm_capabilities import (
    EvidenceSource,
    ParameterDescriptor,
    StructuredPredicate,
)
from cosmic_foundry.computation.decompositions.lu_factorization import LUFactorization
from cosmic_foundry.computation.solvers.coverage import (
    bracketed_scalar_root_predicates,
    constrained_root_predicates,
    directional_derivative_root_predicates,
    fixed_point_root_predicates,
    separable_bracketed_root_predicates,
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
    contraction_bound: float | None

    def __init__(
        self,
        residual: Callable[[Tensor], Tensor],
        fixed_point: Callable[[Tensor], Tensor],
        initial: Tensor,
        contraction_bound: float | None = None,
    ) -> None:
        object.__setattr__(self, "residual", residual)
        object.__setattr__(self, "initial", initial)
        object.__setattr__(self, "equality_constraint_gradients", None)
        object.__setattr__(self, "fixed_point", fixed_point)
        object.__setattr__(self, "contraction_bound", contraction_bound)

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
            fixed_point_contraction_bound=self.contraction_bound,
            matrix_representation_available=False,
            work_budget_fmas=work_budget_fmas,
            memory_budget_bytes=memory_budget_bytes,
            device_kind=device_kind,
            requested_residual_tolerance=requested_residual_tolerance,
            requested_solution_tolerance=requested_solution_tolerance,
        )


class BracketedScalarRootRelation(FiniteDimensionalResidualRelation):
    """Scalar root relation with derivative-free sign-changing bracket evidence."""

    lower: float
    upper: float
    bracket_residual_product_upper_bound: float

    def __init__(
        self,
        residual: Callable[[Tensor], Tensor],
        *,
        lower: float,
        upper: float,
        backend: Any | None = None,
    ) -> None:
        initial: Tensor = Tensor([(lower + upper) / 2.0], backend=backend)
        lower_value = float(residual(Tensor([lower], backend=initial.backend))[0])
        upper_value = float(residual(Tensor([upper], backend=initial.backend))[0])
        product = lower_value * upper_value
        if product > 0.0:
            raise ValueError("bracket endpoints must have opposite residual signs")
        object.__setattr__(self, "residual", residual)
        object.__setattr__(self, "initial", initial)
        object.__setattr__(self, "equality_constraint_gradients", None)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "bracket_residual_product_upper_bound", product)

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
        """Project this bracketed scalar root to primitive solve coordinates."""
        return super().residual_relation_descriptor(
            target_is_zero=True,
            derivative_oracle_kind="none",
            map_linearity_defect=map_linearity_defect,
            map_linearity_evidence=map_linearity_evidence,
            bracket_available=True,
            bracket_residual_product_upper_bound=(
                self.bracket_residual_product_upper_bound
            ),
            matrix_representation_available=False,
            work_budget_fmas=work_budget_fmas,
            memory_budget_bytes=memory_budget_bytes,
            device_kind=device_kind,
            requested_residual_tolerance=requested_residual_tolerance,
            requested_solution_tolerance=requested_solution_tolerance,
        )


class SeparableBracketedRootRelation(FiniteDimensionalResidualRelation):
    """Vector root relation with componentwise sign-changing brackets."""

    lower: tuple[float, ...]
    upper: tuple[float, ...]
    bracket_residual_product_upper_bound: float

    def __init__(
        self,
        residual: Callable[[Tensor], Tensor],
        *,
        lower: tuple[float, ...],
        upper: tuple[float, ...],
        backend: Any | None = None,
    ) -> None:
        if len(lower) != len(upper) or len(lower) < 2:
            raise ValueError("separable bracketed roots require matching vectors")
        initial: Tensor = Tensor(
            [0.5 * (a + b) for a, b in zip(lower, upper, strict=True)],
            backend=backend,
        )
        lower_values = residual(Tensor(list(lower), backend=initial.backend))
        upper_values = residual(Tensor(list(upper), backend=initial.backend))
        products = tuple(
            float(lower_values[index]) * float(upper_values[index])
            for index in range(len(lower))
        )
        product_upper_bound = max(products)
        if product_upper_bound > 0.0:
            raise ValueError("each component bracket must change residual sign")
        object.__setattr__(self, "residual", residual)
        object.__setattr__(self, "initial", initial)
        object.__setattr__(self, "equality_constraint_gradients", None)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(
            self,
            "bracket_residual_product_upper_bound",
            product_upper_bound,
        )

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
        """Project this separable bracketed root to primitive solve coordinates."""
        return super().residual_relation_descriptor(
            target_is_zero=True,
            derivative_oracle_kind="none",
            map_linearity_defect=map_linearity_defect,
            map_linearity_evidence=map_linearity_evidence,
            bracket_available=True,
            bracket_residual_product_upper_bound=(
                self.bracket_residual_product_upper_bound
            ),
            componentwise_separable=True,
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


class BisectionRootSolver:
    """Solve scalar roots using a sign-changing bracket and bisection."""

    root_solver_coverage = bracketed_scalar_root_predicates()

    def __init__(
        self,
        *,
        max_iterations: int = 100,
        tolerance: float = 1e-12,
    ) -> None:
        self._max_iterations = max_iterations
        self._tolerance = tolerance

    def solve(self, relation: BracketedScalarRootRelation) -> Tensor:
        """Return a bracketed scalar root."""
        lower = relation.lower
        upper = relation.upper
        backend = relation.initial.backend
        f_lower = float(relation.residual(Tensor([lower], backend=backend))[0])
        for _ in range(self._max_iterations):
            midpoint = 0.5 * (lower + upper)
            x_mid: Tensor = Tensor([midpoint], backend=backend)
            f_mid = float(relation.residual(x_mid)[0])
            if abs(f_mid) < self._tolerance or abs(upper - lower) < self._tolerance:
                return x_mid
            if f_lower * f_mid <= 0.0:
                upper = midpoint
            else:
                lower = midpoint
                f_lower = f_mid
        return Tensor([0.5 * (lower + upper)], backend=backend)


class SeparableBisectionRootSolver:
    """Solve componentwise separable vector roots by bisection."""

    root_solver_coverage = separable_bracketed_root_predicates()

    def __init__(
        self,
        *,
        max_iterations: int = 100,
        tolerance: float = 1e-12,
    ) -> None:
        self._max_iterations = max_iterations
        self._tolerance = tolerance

    def solve(self, relation: SeparableBracketedRootRelation) -> Tensor:
        """Return a vector root by bisecting each component bracket."""
        backend = relation.initial.backend
        values = list(relation.initial.to_list())
        roots: list[float] = []
        for index, (lower, upper) in enumerate(
            zip(relation.lower, relation.upper, strict=True)
        ):
            values[index] = lower
            f_lower = float(relation.residual(Tensor(values, backend=backend))[index])
            for _ in range(self._max_iterations):
                midpoint = 0.5 * (lower + upper)
                values[index] = midpoint
                f_mid = float(relation.residual(Tensor(values, backend=backend))[index])
                if abs(f_mid) < self._tolerance or abs(upper - lower) < self._tolerance:
                    break
                if f_lower * f_mid <= 0.0:
                    upper = midpoint
                else:
                    lower = midpoint
                    f_lower = f_mid
            roots.append(0.5 * (lower + upper))
            values[index] = roots[-1]
        return Tensor(roots, backend=backend)


__all__ = [
    "BisectionRootSolver",
    "BracketedScalarRootRelation",
    "DirectionalDerivativeRootRelation",
    "FixedPointRootRelation",
    "FixedPointRootSolver",
    "MatrixFreeNewtonKrylovRootSolver",
    "NewtonRootSolver",
    "RootRelation",
    "SeparableBisectionRootSolver",
    "SeparableBracketedRootRelation",
]
