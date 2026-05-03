"""Shared builders for linear-solver coverage regions."""

from __future__ import annotations

from cosmic_foundry.computation.algorithm_capabilities import (
    AffineComparisonPredicate,
    ComparisonPredicate,
    CoverageRegion,
    LinearSolverField,
    MembershipPredicate,
    StructuredPredicate,
)

LINEARITY_TOLERANCE = 1.0e-12
CONDITION_LIMIT = 1.0e8


def coverage(
    owner: type,
    *,
    coverage_predicates: tuple[StructuredPredicate, ...] = (),
) -> CoverageRegion:
    """Return a descriptor-space coverage region owned by ``owner``."""
    return CoverageRegion(
        owner,
        coverage_predicates,
    )


def linear_system_predicates() -> tuple[StructuredPredicate, ...]:
    """Return the primitive predicates for a linear residual solve."""
    return (
        ComparisonPredicate(
            LinearSolverField.MAP_LINEARITY_DEFECT,
            "<=",
            LINEARITY_TOLERANCE,
        ),
        AffineComparisonPredicate(
            {LinearSolverField.DIM_X: 1.0, LinearSolverField.DIM_Y: -1.0},
            "==",
            0.0,
        ),
        MembershipPredicate(
            LinearSolverField.RESIDUAL_TARGET_AVAILABLE,
            frozenset({True}),
        ),
        MembershipPredicate(
            LinearSolverField.ACCEPTANCE_RELATION,
            frozenset({"residual_below_tolerance"}),
        ),
        MembershipPredicate(
            LinearSolverField.OBJECTIVE_RELATION,
            frozenset({"none"}),
        ),
    )


def dense_matrix_predicates() -> tuple[MembershipPredicate, ...]:
    """Return predicates for an available dense assembled matrix."""
    return (
        MembershipPredicate(
            LinearSolverField.MATRIX_REPRESENTATION_AVAILABLE,
            frozenset({True}),
        ),
        MembershipPredicate(
            LinearSolverField.LINEAR_OPERATOR_MATRIX_AVAILABLE,
            frozenset({True}),
        ),
    )


def budget_predicates() -> tuple[AffineComparisonPredicate, ...]:
    """Return conservative work and memory budget predicates."""
    return (
        AffineComparisonPredicate(
            {
                LinearSolverField.WORK_BUDGET_FMAS: 1.0,
                LinearSolverField.ASSEMBLY_COST_FMAS: -1.0,
            },
            ">=",
            0.0,
        ),
        AffineComparisonPredicate(
            {
                LinearSolverField.WORK_BUDGET_FMAS: 1.0,
                LinearSolverField.MATVEC_COST_FMAS: -1.0,
            },
            ">=",
            0.0,
        ),
        AffineComparisonPredicate(
            {
                LinearSolverField.MEMORY_BUDGET_BYTES: 1.0,
                LinearSolverField.LINEAR_OPERATOR_MEMORY_BYTES: -1.0,
            },
            ">=",
            0.0,
        ),
    )


def matrix_free_operator_predicates() -> tuple[MembershipPredicate, ...]:
    """Return predicates for solvers that use operator application directly."""
    return (
        MembershipPredicate(
            LinearSolverField.MATRIX_REPRESENTATION_AVAILABLE,
            frozenset({False}),
        ),
        MembershipPredicate(
            LinearSolverField.LINEAR_OPERATOR_MATRIX_AVAILABLE,
            frozenset({False}),
        ),
        MembershipPredicate(
            LinearSolverField.OPERATOR_APPLICATION_AVAILABLE,
            frozenset({True}),
        ),
    )


__all__: list[str] = []
