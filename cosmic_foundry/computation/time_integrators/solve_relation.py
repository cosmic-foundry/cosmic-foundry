"""Solve-relation descriptors for time-integration steps."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from cosmic_foundry.computation.algorithm_capabilities import (
    EvidenceSource,
    LinearOperatorEvidence,
    ParameterDescriptor,
    SolveRelationField,
    TransformationRelation,
    TransformationSpace,
    assembled_linear_transformation_relation,
    transformation_relation_coordinates,
)
from cosmic_foundry.computation.tensor import Tensor


@runtime_checkable
class AffineRHSProtocol(Protocol):
    """ODE RHS that exposes the exact linear operator of an affine map."""

    def __call__(self, t: float, u: Tensor) -> Tensor:
        """Evaluate ``f(t, u)``."""
        ...

    def linear_operator(self, t: float, u: Tensor) -> Tensor:
        """Return the matrix A in ``f(t, u) = A u + b``."""
        ...


def time_integrator_step_solve_relation_descriptor(
    integrator: Any,
    rhs: Any,
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
    upper-triangular entries induce implicit stage equations.  When the RHS
    exposes an exact affine operator, those stage equations compose to an
    affine residual; otherwise the descriptor records unknown linearity with
    Jacobian-callback evidence.
    """
    if dt <= 0.0:
        raise ValueError("time-step solve relations require dt > 0")
    if len(state.u.shape) != 1 or state.u.shape[0] <= 0:
        raise ValueError("time-step solve relations require a nonempty state vector")
    stage_matrix = getattr(integrator, "A_sym", ())
    if not stage_matrix:
        raise ValueError("time-step solve relations require stage equations")
    equality_constraint_count = _active_equality_constraint_count(rhs, state)

    if _stage_matrix_is_strictly_lower(stage_matrix):
        relation = _transformation_relation(
            state.u,
            variable_count=state.u.shape[0],
            equality_constraint_count=equality_constraint_count,
            map_linearity_defect=0.0,
            map_linearity_evidence="exact",
            matrix_representation_available=True,
            derivative_oracle_kind="matrix",
            work_budget_fmas=work_budget_fmas,
            memory_budget_bytes=memory_budget_bytes,
            device_kind=device_kind,
        )
        return _descriptor_from_relation(
            relation,
            requested_residual_tolerance=requested_residual_tolerance,
            requested_solution_tolerance=requested_solution_tolerance,
        )
    if not _stage_matrix_has_implicit_coupling(stage_matrix):
        raise ValueError("time-step solve relation has no stage-equation premise")
    affine_rhs = isinstance(rhs, AffineRHSProtocol)
    evidence: tuple[LinearOperatorEvidence, ...] = ()
    relation = _transformation_relation(
        state.u,
        variable_count=state.u.shape[0] * len(stage_matrix),
        equality_constraint_count=equality_constraint_count,
        map_linearity_defect=None,
        map_linearity_evidence="unavailable",
        matrix_representation_available=False,
        derivative_oracle_kind="jacobian_callback",
        work_budget_fmas=work_budget_fmas,
        memory_budget_bytes=memory_budget_bytes,
        device_kind=device_kind,
    )
    if affine_rhs:
        stage_times = _stage_times(integrator, state.t, dt, len(stage_matrix))
        if len(stage_times) != len(stage_matrix):
            raise ValueError("stage time nodes must match the implicit stage matrix")
        linear_operator = _AffineStageResidualOperator(
            rhs, stage_matrix, stage_times, state.u, dt
        )
        evidence = (
            LinearOperatorEvidence(
                linear_operator,
                linear_operator.rhs(),
                (),
            ),
        )
        relation = assembled_linear_transformation_relation(
            evidence[0],
            equality_constraint_count=equality_constraint_count,
            work_budget_fmas=work_budget_fmas,
            memory_budget_bytes=memory_budget_bytes,
            device_kind=device_kind,
        )
    return _descriptor_from_relation(
        relation,
        requested_residual_tolerance=requested_residual_tolerance,
        requested_solution_tolerance=requested_solution_tolerance,
        evidence=evidence,
    )


def _transformation_relation(
    state_vector: Tensor,
    *,
    variable_count: int,
    equality_constraint_count: int,
    map_linearity_defect: float | None,
    map_linearity_evidence: EvidenceSource,
    matrix_representation_available: bool,
    derivative_oracle_kind: str,
    work_budget_fmas: float,
    memory_budget_bytes: float,
    device_kind: str,
) -> TransformationRelation:
    space = TransformationSpace(
        variable_count,
        _backend_kind(state_vector),
        device_kind,
    )
    return TransformationRelation(
        domain=space,
        codomain=space,
        residual_target_available=True,
        target_is_zero=True,
        map_linearity_defect=map_linearity_defect,
        map_linearity_evidence=map_linearity_evidence,
        matrix_representation_available=matrix_representation_available,
        operator_application_available=True,
        derivative_oracle_kind=derivative_oracle_kind,
        equality_constraint_count=equality_constraint_count,
        objective_relation="none",
        acceptance_relation="residual_below_tolerance",
        backend_kind=space.backend_kind,
        device_kind=device_kind,
        work_budget_fmas=work_budget_fmas,
        memory_budget_bytes=memory_budget_bytes,
    )


def _descriptor_from_relation(
    relation: TransformationRelation,
    *,
    requested_residual_tolerance: float,
    requested_solution_tolerance: float,
    evidence: tuple[LinearOperatorEvidence, ...] = (),
) -> ParameterDescriptor:
    return ParameterDescriptor(
        transformation_relation_coordinates(
            relation,
            frozenset(SolveRelationField),
            requested_residual_tolerance=requested_residual_tolerance,
            requested_solution_tolerance=requested_solution_tolerance,
        ),
        evidence=evidence,
    )


def _active_equality_constraint_count(rhs: Any, state: Any) -> int:
    active = getattr(state, "active_constraints", None) or frozenset()
    count = len(active)
    if count == 0:
        return 0
    gradients = getattr(rhs, "constraint_gradients", None)
    if not callable(gradients):
        raise ValueError(
            "active equality constraints require RHS constraint-gradient evidence"
        )
    return count


class _AffineStageResidualOperator:
    """Block operator for implicit RK stage equations with affine RHS."""

    def __init__(
        self,
        rhs: AffineRHSProtocol,
        stage_matrix: Any,
        stage_times: tuple[float, ...],
        state_vector: Tensor,
        dt: float,
    ) -> None:
        self._rhs = rhs
        self._stage_matrix = tuple(
            tuple(float(entry) for entry in row) for row in stage_matrix
        )
        self._stage_times = stage_times
        self._state_vector = state_vector
        self._dt = dt
        self._stage_count = len(stage_times)
        self._state_dimension = state_vector.shape[0]

    def apply(self, values: Tensor) -> Tensor:
        if values.shape != (self._stage_count * self._state_dimension,):
            raise ValueError("stage residual operator received wrong vector shape")
        blocks = self._split(values)
        result: list[float] = []
        for row_index in range(self._stage_count):
            row = [
                float(blocks[row_index][component])
                for component in range(self._state_dimension)
            ]
            for column_index in range(self._stage_count):
                coefficient = self._stage_matrix[row_index][column_index]
                if coefficient == 0.0:
                    continue
                operator = self._rhs.linear_operator(
                    self._stage_times[column_index],
                    blocks[column_index],
                )
                contribution = operator @ blocks[column_index]
                for component in range(self._state_dimension):
                    row[component] -= (
                        self._dt * coefficient * float(contribution[component])
                    )
            result.extend(row)
        return Tensor(result, backend=values.backend)

    def rhs(self) -> Tensor:
        offsets = tuple(
            self._rhs(self._stage_times[index], self._state_vector)
            - self._rhs.linear_operator(self._stage_times[index], self._state_vector)
            @ self._state_vector
            for index in range(self._stage_count)
        )
        result: list[float] = []
        for row_index in range(self._stage_count):
            row = [
                float(self._state_vector[component])
                for component in range(self._state_dimension)
            ]
            for column_index in range(self._stage_count):
                coefficient = self._stage_matrix[row_index][column_index]
                if coefficient == 0.0:
                    continue
                for component in range(self._state_dimension):
                    row[component] += (
                        self._dt * coefficient * float(offsets[column_index][component])
                    )
            result.extend(row)
        return Tensor(result, backend=self._state_vector.backend)

    def _split(self, values: Tensor) -> tuple[Tensor, ...]:
        return tuple(
            values[stage * self._state_dimension : (stage + 1) * self._state_dimension]
            for stage in range(self._stage_count)
        )


def _stage_times(
    integrator: Any,
    time: float,
    dt: float,
    stage_count: int,
) -> tuple[float, ...]:
    nodes = getattr(integrator, "c_sym", None)
    if nodes is None:
        nodes = (0.0,) * stage_count
    return tuple(time + dt * float(node) for node in nodes)


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


__all__ = [
    "AffineRHSProtocol",
    "time_integrator_step_solve_relation_descriptor",
]
