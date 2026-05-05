"""Newton solver for finite-dimensional root relations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from cosmic_foundry.computation.algorithm_capabilities import (
    EvidenceSource,
    ParameterDescriptor,
    SolveRelationField,
    StructuredPredicate,
    TransformationRelation,
    TransformationSpace,
    transformation_relation_coordinates,
)
from cosmic_foundry.computation.decompositions.lu_factorization import LUFactorization
from cosmic_foundry.computation.solvers.coverage import nonlinear_root_predicates
from cosmic_foundry.computation.tensor import Tensor, einsum, norm


@dataclass(frozen=True)
class RootSolveProblem:
    """Finite-dimensional residual relation ``F(x) = 0``."""

    residual: Callable[[Tensor], Tensor]
    jacobian: Callable[[Tensor], Tensor]
    initial: Tensor
    equality_constraint_gradients: Tensor | None = None

    @property
    def equality_constraint_count(self) -> int:
        """Return the number of equality constraints active in the relation."""
        if self.equality_constraint_gradients is None:
            return 0
        return self.equality_constraint_gradients.shape[0]

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
        """Project this root problem to primitive solve-relation coordinates."""
        domain = TransformationSpace(
            self.initial.shape[0],
            _backend_kind(self.initial),
            device_kind,
        )
        codomain = TransformationSpace(
            self.residual(self.initial).shape[0],
            domain.backend_kind,
            device_kind,
        )
        relation = TransformationRelation(
            domain=domain,
            codomain=codomain,
            residual_target_available=True,
            target_is_zero=True,
            map_linearity_defect=map_linearity_defect,
            map_linearity_evidence=map_linearity_evidence,
            matrix_representation_available=False,
            operator_application_available=True,
            derivative_oracle_kind="jacobian_callback",
            equality_constraint_count=self.equality_constraint_count,
            objective_relation="none",
            acceptance_relation="residual_below_tolerance",
            backend_kind=domain.backend_kind,
            device_kind=device_kind,
            work_budget_fmas=work_budget_fmas,
            memory_budget_bytes=memory_budget_bytes,
        )
        return ParameterDescriptor(
            transformation_relation_coordinates(
                relation,
                frozenset(SolveRelationField),
                requested_residual_tolerance=requested_residual_tolerance,
                requested_solution_tolerance=requested_solution_tolerance,
            )
        )


class NewtonRootSolver:
    """Solve ``F(x) = 0`` by Newton iteration."""

    root_solver_coverage: tuple[tuple[StructuredPredicate, ...], ...] = (
        nonlinear_root_predicates()
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
        problem: RootSolveProblem,
    ) -> Tensor:
        """Return a root of the supplied residual relation."""
        x = problem.initial
        gram: Tensor | None = None
        if problem.equality_constraint_gradients is not None:
            gradients = problem.equality_constraint_gradients
            gram = einsum("ij,kj->ik", gradients, gradients)
        for _ in range(self._max_iterations):
            Fx = problem.residual(x)
            if self._small(Fx, x):
                break
            delta = self._lu.factorize(problem.jacobian(x)).solve(
                Tensor.zeros(x.shape[0], backend=x.backend) - Fx
            )
            if problem.equality_constraint_gradients is not None and gram is not None:
                gradients = problem.equality_constraint_gradients
                xi = self._lu.factorize(gram).solve(gradients @ delta)
                delta = delta - einsum("ij,i->j", gradients, xi)
            x = x + delta
            if self._small(delta, x):
                break
        return x

    def _small(self, vector: Tensor, scale: Tensor) -> bool:
        return float(norm(vector)) < self._tolerance * (1.0 + float(norm(scale)))


def _backend_kind(tensor: Tensor) -> str:
    name = type(tensor.backend).__name__.lower()
    if "numpy" in name:
        return "numpy"
    if "jax" in name:
        return "jax"
    if "python" in name:
        return "python"
    return "unknown"


__all__ = ["NewtonRootSolver", "RootSolveProblem"]
