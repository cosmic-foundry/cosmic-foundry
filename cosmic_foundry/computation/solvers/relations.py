"""Finite-dimensional solve relations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from cosmic_foundry.computation.algorithm_capabilities import (
    EvidenceSource,
    ParameterDescriptor,
    SolveRelationField,
    TransformationRelation,
    TransformationSpace,
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


def _backend_kind(tensor: Tensor) -> str:
    name = type(tensor.backend).__name__.lower()
    if "numpy" in name:
        return "numpy"
    if "jax" in name:
        return "jax"
    if "python" in name:
        return "python"
    return "unknown"


__all__ = ["FiniteDimensionalResidualRelation"]
