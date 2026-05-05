"""Shared builders for linear-solver coverage regions."""

from __future__ import annotations

from cosmic_foundry.computation.algorithm_capabilities import (
    CONDITION_LIMIT as CONDITION_LIMIT,
)
from cosmic_foundry.computation.algorithm_capabilities import (
    LINEARITY_TOLERANCE as LINEARITY_TOLERANCE,
)
from cosmic_foundry.computation.algorithm_capabilities import (
    AffineComparisonPredicate,
    ComparisonPredicate,
    CoverageRegion,
    EvidencePredicate,
    LinearSolverField,
    MembershipPredicate,
    SolveRelationField,
    StructuredPredicate,
)


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
            SolveRelationField.MAP_LINEARITY_DEFECT,
            "<=",
            LINEARITY_TOLERANCE,
        ),
        AffineComparisonPredicate(
            {SolveRelationField.DIM_X: 1.0, SolveRelationField.DIM_Y: -1.0},
            "==",
            0.0,
        ),
        MembershipPredicate(
            SolveRelationField.RESIDUAL_TARGET_AVAILABLE,
            frozenset({True}),
        ),
        MembershipPredicate(
            SolveRelationField.ACCEPTANCE_RELATION,
            frozenset({"residual_below_tolerance"}),
        ),
        MembershipPredicate(
            SolveRelationField.OBJECTIVE_RELATION,
            frozenset({"none"}),
        ),
    )


def least_squares_predicates() -> tuple[StructuredPredicate, ...]:
    """Return primitive predicates for a linear least-squares objective."""
    return (
        ComparisonPredicate(
            SolveRelationField.MAP_LINEARITY_DEFECT,
            "<=",
            LINEARITY_TOLERANCE,
        ),
        MembershipPredicate(
            SolveRelationField.RESIDUAL_TARGET_AVAILABLE,
            frozenset({True}),
        ),
        MembershipPredicate(
            SolveRelationField.OBJECTIVE_RELATION,
            frozenset({"least_squares"}),
        ),
        MembershipPredicate(
            SolveRelationField.ACCEPTANCE_RELATION,
            frozenset({"objective_minimum"}),
        ),
        MembershipPredicate(
            SolveRelationField.MATRIX_REPRESENTATION_AVAILABLE,
            frozenset({True}),
        ),
    )


def dense_matrix_predicates() -> tuple[MembershipPredicate, ...]:
    """Return predicates for an available dense assembled matrix."""
    return (
        MembershipPredicate(
            SolveRelationField.MATRIX_REPRESENTATION_AVAILABLE,
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
                SolveRelationField.WORK_BUDGET_FMAS: 1.0,
                LinearSolverField.ASSEMBLY_COST_FMAS: -1.0,
            },
            ">=",
            0.0,
        ),
        AffineComparisonPredicate(
            {
                SolveRelationField.WORK_BUDGET_FMAS: 1.0,
                LinearSolverField.MATVEC_COST_FMAS: -1.0,
            },
            ">=",
            0.0,
        ),
        AffineComparisonPredicate(
            {
                SolveRelationField.MEMORY_BUDGET_BYTES: 1.0,
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
            SolveRelationField.MATRIX_REPRESENTATION_AVAILABLE,
            frozenset({False}),
        ),
        MembershipPredicate(
            LinearSolverField.LINEAR_OPERATOR_MATRIX_AVAILABLE,
            frozenset({False}),
        ),
        MembershipPredicate(
            SolveRelationField.OPERATOR_APPLICATION_AVAILABLE,
            frozenset({True}),
        ),
    )


def nonlinear_root_predicates(
    *,
    derivative_oracle_kind: str = "jacobian_callback",
    equality_constraint_predicates: tuple[StructuredPredicate, ...] = (),
) -> tuple[tuple[StructuredPredicate, ...], ...]:
    """Return primitive predicate sets for nonlinear residual roots."""
    common: tuple[StructuredPredicate, ...] = (
        MembershipPredicate(
            SolveRelationField.TARGET_IS_ZERO,
            frozenset({True}),
        ),
        MembershipPredicate(
            SolveRelationField.ACCEPTANCE_RELATION,
            frozenset({"residual_below_tolerance"}),
        ),
        MembershipPredicate(
            SolveRelationField.DERIVATIVE_ORACLE_KIND,
            frozenset({derivative_oracle_kind}),
        ),
        AffineComparisonPredicate(
            {SolveRelationField.DIM_X: 1.0, SolveRelationField.DIM_Y: -1.0},
            "==",
            0.0,
        ),
        *equality_constraint_predicates,
    )
    return (
        (
            ComparisonPredicate(
                SolveRelationField.MAP_LINEARITY_DEFECT,
                ">",
                LINEARITY_TOLERANCE,
            ),
            *common,
        ),
        (
            EvidencePredicate(
                SolveRelationField.MAP_LINEARITY_DEFECT,
                frozenset({"unavailable"}),
            ),
            *common,
        ),
    )


def unconstrained_root_predicates() -> tuple[tuple[StructuredPredicate, ...], ...]:
    """Return predicate sets for nonlinear roots without equality constraints."""
    return nonlinear_root_predicates(
        equality_constraint_predicates=(
            ComparisonPredicate(SolveRelationField.EQUALITY_CONSTRAINT_COUNT, "==", 0),
        )
    )


def constrained_root_predicates() -> tuple[tuple[StructuredPredicate, ...], ...]:
    """Return predicate sets for nonlinear roots with equality constraints."""
    return nonlinear_root_predicates(
        equality_constraint_predicates=(
            ComparisonPredicate(SolveRelationField.EQUALITY_CONSTRAINT_COUNT, ">", 0),
        )
    )


def directional_derivative_root_predicates() -> (
    tuple[tuple[StructuredPredicate, ...], ...]
):
    """Return predicate sets for unconstrained nonlinear roots with JVP evidence."""
    return nonlinear_root_predicates(
        derivative_oracle_kind="jvp",
        equality_constraint_predicates=(
            ComparisonPredicate(SolveRelationField.EQUALITY_CONSTRAINT_COUNT, "==", 0),
        ),
    )


def fixed_point_root_predicates() -> tuple[tuple[StructuredPredicate, ...], ...]:
    """Return predicate sets for unconstrained roots with iteration-map evidence."""
    return nonlinear_root_predicates(
        derivative_oracle_kind="fixed_point_map",
        equality_constraint_predicates=(
            ComparisonPredicate(SolveRelationField.EQUALITY_CONSTRAINT_COUNT, "==", 0),
            ComparisonPredicate(
                SolveRelationField.FIXED_POINT_CONTRACTION_BOUND,
                "<",
                1.0,
            ),
        ),
    )


__all__ = [
    "CONDITION_LIMIT",
    "LINEARITY_TOLERANCE",
    "budget_predicates",
    "coverage",
    "dense_matrix_predicates",
    "directional_derivative_root_predicates",
    "fixed_point_root_predicates",
    "least_squares_predicates",
    "linear_system_predicates",
    "matrix_free_operator_predicates",
    "constrained_root_predicates",
    "nonlinear_root_predicates",
    "unconstrained_root_predicates",
]
