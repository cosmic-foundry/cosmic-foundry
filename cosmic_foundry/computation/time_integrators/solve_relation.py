"""Solve-relation descriptors for time-integration steps."""

from __future__ import annotations

from typing import Any

from cosmic_foundry.computation.algorithm_capabilities import (
    DescriptorCoordinate,
    EvidenceSource,
    ParameterDescriptor,
    SolveRelationField,
)
from cosmic_foundry.computation.tensor import Tensor


def time_integrator_step_solve_relation_descriptor(
    integrator: Any,
    state: Any,
    dt: float,
    *,
    requested_residual_tolerance: float = 1.0e-8,
    requested_solution_tolerance: float = 1.0e-8,
    work_budget_fmas: float = 1.0e9,
    memory_budget_bytes: float = 1.0e9,
    device_kind: str = "cpu",
) -> ParameterDescriptor:
    """Return the solve relation induced by one time-integration step.

    Strictly explicit stage matrices induce an affine next-state residual
    ``R(x) = x - Phi_h(t_n, u_n)``.  Stage matrices with on-diagonal or
    upper-triangular entries induce implicit stage equations; without a
    symbolic linearity certificate for the RHS, those stage equations are
    nonlinear-root solve relations with Jacobian-callback evidence.
    """
    if dt <= 0.0:
        raise ValueError("time-step solve relations require dt > 0")
    if len(state.u.shape) != 1 or state.u.shape[0] <= 0:
        raise ValueError("time-step solve relations require a nonempty state vector")
    stage_matrix = getattr(integrator, "A_sym", ())
    if not stage_matrix:
        raise ValueError("time-step solve relations require stage equations")

    if _stage_matrix_is_strictly_lower(stage_matrix):
        return _descriptor(
            state.u,
            variable_count=state.u.shape[0],
            map_linearity_defect=0.0,
            map_linearity_evidence="exact",
            matrix_representation_available=True,
            derivative_oracle_kind="matrix",
            requested_residual_tolerance=requested_residual_tolerance,
            requested_solution_tolerance=requested_solution_tolerance,
            work_budget_fmas=work_budget_fmas,
            memory_budget_bytes=memory_budget_bytes,
            device_kind=device_kind,
        )
    if not _stage_matrix_has_implicit_coupling(stage_matrix):
        raise ValueError("time-step solve relation has no stage-equation premise")
    return _descriptor(
        state.u,
        variable_count=state.u.shape[0] * len(stage_matrix),
        map_linearity_defect=None,
        map_linearity_evidence="unavailable",
        matrix_representation_available=False,
        derivative_oracle_kind="jacobian_callback",
        requested_residual_tolerance=requested_residual_tolerance,
        requested_solution_tolerance=requested_solution_tolerance,
        work_budget_fmas=work_budget_fmas,
        memory_budget_bytes=memory_budget_bytes,
        device_kind=device_kind,
    )


def _descriptor(
    state_vector: Tensor,
    *,
    variable_count: int,
    map_linearity_defect: float | None,
    map_linearity_evidence: EvidenceSource,
    matrix_representation_available: bool,
    derivative_oracle_kind: str,
    requested_residual_tolerance: float,
    requested_solution_tolerance: float,
    work_budget_fmas: float,
    memory_budget_bytes: float,
    device_kind: str,
) -> ParameterDescriptor:
    field = SolveRelationField
    return ParameterDescriptor(
        {
            field.DIM_X: DescriptorCoordinate(variable_count),
            field.DIM_Y: DescriptorCoordinate(variable_count),
            field.AUXILIARY_SCALAR_COUNT: DescriptorCoordinate(0),
            field.EQUALITY_CONSTRAINT_COUNT: DescriptorCoordinate(0),
            field.NORMALIZATION_CONSTRAINT_COUNT: DescriptorCoordinate(0),
            field.RESIDUAL_TARGET_AVAILABLE: DescriptorCoordinate(True),
            field.TARGET_IS_ZERO: DescriptorCoordinate(True),
            field.MAP_LINEARITY_DEFECT: DescriptorCoordinate(
                map_linearity_defect,
                evidence=map_linearity_evidence,
            ),
            field.MATRIX_REPRESENTATION_AVAILABLE: DescriptorCoordinate(
                matrix_representation_available
            ),
            field.OPERATOR_APPLICATION_AVAILABLE: DescriptorCoordinate(True),
            field.DERIVATIVE_ORACLE_KIND: DescriptorCoordinate(derivative_oracle_kind),
            field.OBJECTIVE_RELATION: DescriptorCoordinate("none"),
            field.ACCEPTANCE_RELATION: DescriptorCoordinate("residual_below_tolerance"),
            field.REQUESTED_RESIDUAL_TOLERANCE: DescriptorCoordinate(
                requested_residual_tolerance
            ),
            field.REQUESTED_SOLUTION_TOLERANCE: DescriptorCoordinate(
                requested_solution_tolerance
            ),
            field.BACKEND_KIND: DescriptorCoordinate(_backend_kind(state_vector)),
            field.DEVICE_KIND: DescriptorCoordinate(device_kind),
            field.WORK_BUDGET_FMAS: DescriptorCoordinate(work_budget_fmas),
            field.MEMORY_BUDGET_BYTES: DescriptorCoordinate(memory_budget_bytes),
        }
    )


def _stage_matrix_is_strictly_lower(matrix: Any) -> bool:
    return bool(matrix) and all(
        entry == 0
        for row_index, row in enumerate(matrix)
        for column_index, entry in enumerate(row)
        if column_index >= row_index
    )


def _stage_matrix_has_implicit_coupling(matrix: Any) -> bool:
    return bool(matrix) and any(
        entry != 0
        for row_index, row in enumerate(matrix)
        for column_index, entry in enumerate(row)
        if column_index >= row_index
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


__all__ = ["time_integrator_step_solve_relation_descriptor"]
