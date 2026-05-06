"""Finite-dimensional solve relations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from cosmic_foundry.computation.algorithm_capabilities import (
    DecompositionField,
    EvidenceSource,
    LinearOperatorEvidence,
    ParameterDescriptor,
    SolveRelationField,
    TransformationRelation,
    TransformationSpace,
    assembled_linear_transformation_relation,
    decomposition_coordinates,
    transformation_relation_coordinates,
)
from cosmic_foundry.computation.tensor import Tensor


@dataclass(frozen=True)
class FiniteDimensionalResidualRelation:
    """Finite-dimensional residual map with an initial point."""

    residual: Callable[[Tensor], Tensor]
    initial: Tensor
    equality_constraint_gradients: Tensor | None = None

    @property
    def equality_constraint_count(self) -> int:
        """Return the number of equality constraints active in the relation."""
        if self.equality_constraint_gradients is None:
            return 0
        return self.equality_constraint_gradients.shape[0]

    def residual_relation_descriptor(
        self,
        *,
        target_is_zero: bool,
        derivative_oracle_kind: str,
        map_linearity_defect: float | None = None,
        map_linearity_evidence: EvidenceSource = "unavailable",
        fixed_point_contraction_bound: float | None = None,
        bracket_available: bool = False,
        bracket_residual_product_upper_bound: float | None = None,
        componentwise_separable: bool = False,
        matrix_representation_available: bool = False,
        requested_residual_tolerance: float = 1.0e-8,
        requested_solution_tolerance: float = 1.0e-8,
        work_budget_fmas: float = 1.0e9,
        memory_budget_bytes: float = 1.0e9,
        device_kind: str = "cpu",
    ) -> ParameterDescriptor:
        """Project this residual relation to primitive solve-relation coordinates."""
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
            target_is_zero=target_is_zero,
            map_linearity_defect=map_linearity_defect,
            map_linearity_evidence=map_linearity_evidence,
            fixed_point_contraction_bound=fixed_point_contraction_bound,
            bracket_available=bracket_available,
            bracket_residual_product_upper_bound=(bracket_residual_product_upper_bound),
            componentwise_separable=componentwise_separable,
            matrix_representation_available=matrix_representation_available,
            operator_application_available=True,
            derivative_oracle_kind=derivative_oracle_kind,
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


class LinearResidualRelation(FiniteDimensionalResidualRelation):
    """Finite-dimensional residual relation with assembled linear evidence."""

    def __init__(
        self,
        evidence: LinearOperatorEvidence,
        *,
        equality_constraint_count: int = 0,
    ) -> None:
        self.linear_operator_evidence = evidence
        self._equality_constraint_count = equality_constraint_count
        _rows, columns = _linear_evidence_shape(evidence)
        initial = Tensor.zeros(columns, backend=evidence.rhs.backend)

        def residual(x: Tensor) -> Tensor:
            return evidence.operator.apply(x) - evidence.rhs

        super().__init__(residual, initial)

    @property
    def equality_constraint_count(self) -> int:
        """Return the number of equality constraints active in the relation."""
        return self._equality_constraint_count

    def decomposition_descriptor(
        self,
        *,
        factorization_work_budget_fmas: float = 1.0e9,
        factorization_memory_budget_bytes: float = 1.0e9,
    ) -> ParameterDescriptor:
        """Project this relation's assembled matrix evidence to decomposition space."""
        return ParameterDescriptor(
            decomposition_coordinates(
                self.linear_operator_evidence,
                frozenset(DecompositionField),
                factorization_work_budget_fmas=factorization_work_budget_fmas,
                factorization_memory_budget_bytes=factorization_memory_budget_bytes,
            ),
            evidence=(self.linear_operator_evidence,),
        )

    def solve_relation_descriptor(
        self,
        *,
        requested_residual_tolerance: float = 1.0e-8,
        requested_solution_tolerance: float = 1.0e-8,
        work_budget_fmas: float = 1.0e9,
        memory_budget_bytes: float = 1.0e9,
        device_kind: str = "cpu",
    ) -> ParameterDescriptor:
        """Project this linear residual relation to primitive solve coordinates."""
        relation = assembled_linear_transformation_relation(
            self.linear_operator_evidence,
            equality_constraint_count=self.equality_constraint_count,
            work_budget_fmas=work_budget_fmas,
            memory_budget_bytes=memory_budget_bytes,
            device_kind=device_kind,
        )
        return ParameterDescriptor(
            transformation_relation_coordinates(
                relation,
                frozenset(SolveRelationField),
                requested_residual_tolerance=requested_residual_tolerance,
                requested_solution_tolerance=requested_solution_tolerance,
            ),
            evidence=(self.linear_operator_evidence,),
        )


class LeastSquaresRelation(LinearResidualRelation):
    """Linear residual relation solved by minimizing the residual norm."""

    def solve_relation_descriptor(
        self,
        *,
        requested_residual_tolerance: float = 1.0e-8,
        requested_solution_tolerance: float = 1.0e-8,
        work_budget_fmas: float = 1.0e9,
        memory_budget_bytes: float = 1.0e9,
        device_kind: str = "cpu",
    ) -> ParameterDescriptor:
        """Project this least-squares relation to primitive solve coordinates."""
        rows, columns = _linear_evidence_shape(self.linear_operator_evidence)
        backend_kind = _backend_kind(self.linear_operator_evidence.rhs)
        relation = TransformationRelation(
            domain=TransformationSpace(columns, backend_kind, device_kind),
            codomain=TransformationSpace(rows, backend_kind, device_kind),
            residual_target_available=True,
            target_is_zero=False,
            map_linearity_defect=0.0,
            map_linearity_evidence="exact",
            matrix_representation_available=True,
            operator_application_available=True,
            derivative_oracle_kind="matrix",
            equality_constraint_count=self.equality_constraint_count,
            objective_relation="least_squares",
            acceptance_relation="objective_minimum",
            backend_kind=backend_kind,
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
            ),
            evidence=(self.linear_operator_evidence,),
        )


def _linear_evidence_shape(evidence: LinearOperatorEvidence) -> tuple[int, int]:
    matrix = evidence.matrix
    rows = len(matrix) if matrix else evidence.rhs.shape[0]
    columns = len(matrix[0]) if matrix else evidence.rhs.shape[0]
    if evidence.rhs.shape[0] != rows or any(len(row) != columns for row in matrix):
        raise ValueError("linear residual evidence has inconsistent matrix/RHS shape")
    return rows, columns


def _backend_kind(tensor: Tensor) -> str:
    name = type(tensor.backend).__name__.lower()
    if "numpy" in name:
        return "numpy"
    if "jax" in name:
        return "jax"
    if "python" in name:
        return "python"
    return "unknown"


__all__ = [
    "FiniteDimensionalResidualRelation",
    "LeastSquaresRelation",
    "LinearResidualRelation",
]
