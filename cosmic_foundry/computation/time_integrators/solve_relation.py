"""Solve-relation descriptors for time-integration steps."""

from __future__ import annotations

from typing import Any

from cosmic_foundry.computation.algorithm_capabilities import (
    DescriptorCoordinate,
    LinearSolverField,
    ParameterDescriptor,
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
    """Return the solve relation for an explicit one-step state update.

    The unknown is the next state x.  For a strictly explicit stage matrix,
    every stage value is known before x is formed, so the residual relation is
    R(x) = x - Phi_h(t_n, u_n) = 0.  That relation is affine in x with identity
    derivative; no nonlinear or coupled stage solve is being claimed here.
    """
    if dt <= 0.0:
        raise ValueError("time-step solve relations require dt > 0")
    if len(state.u.shape) != 1 or state.u.shape[0] <= 0:
        raise ValueError("time-step solve relations require a nonempty state vector")
    if not _stage_matrix_is_strictly_lower(getattr(integrator, "A_sym", ())):
        raise ValueError("time-step solve relation is explicit-stage only")

    field = LinearSolverField
    n = state.u.shape[0]
    return ParameterDescriptor(
        {
            field.DIM_X: DescriptorCoordinate(n),
            field.DIM_Y: DescriptorCoordinate(n),
            field.AUXILIARY_SCALAR_COUNT: DescriptorCoordinate(0),
            field.EQUALITY_CONSTRAINT_COUNT: DescriptorCoordinate(0),
            field.NORMALIZATION_CONSTRAINT_COUNT: DescriptorCoordinate(0),
            field.RESIDUAL_TARGET_AVAILABLE: DescriptorCoordinate(True),
            field.TARGET_IS_ZERO: DescriptorCoordinate(True),
            field.MAP_LINEARITY_DEFECT: DescriptorCoordinate(0.0),
            field.MATRIX_REPRESENTATION_AVAILABLE: DescriptorCoordinate(True),
            field.OPERATOR_APPLICATION_AVAILABLE: DescriptorCoordinate(True),
            field.DERIVATIVE_ORACLE_KIND: DescriptorCoordinate("matrix"),
            field.OBJECTIVE_RELATION: DescriptorCoordinate("none"),
            field.ACCEPTANCE_RELATION: DescriptorCoordinate("residual_below_tolerance"),
            field.REQUESTED_RESIDUAL_TOLERANCE: DescriptorCoordinate(
                requested_residual_tolerance
            ),
            field.REQUESTED_SOLUTION_TOLERANCE: DescriptorCoordinate(
                requested_solution_tolerance
            ),
            field.BACKEND_KIND: DescriptorCoordinate(_backend_kind(state.u)),
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
