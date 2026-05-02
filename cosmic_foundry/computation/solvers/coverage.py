"""Shared builders for linear-solver coverage regions."""

from __future__ import annotations

from cosmic_foundry.computation.algorithm_capabilities import (
    AffineComparisonPredicate,
    ComparisonPredicate,
    CoverageRegion,
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
        ComparisonPredicate("map_linearity_defect", "<=", LINEARITY_TOLERANCE),
        AffineComparisonPredicate({"dim_x": 1.0, "dim_y": -1.0}, "==", 0.0),
        MembershipPredicate("residual_target_available", frozenset({True})),
        MembershipPredicate(
            "acceptance_relation",
            frozenset({"residual_below_tolerance"}),
        ),
    )


def dense_matrix_predicates() -> tuple[MembershipPredicate, ...]:
    """Return predicates for an available dense assembled matrix."""
    return (
        MembershipPredicate("matrix_representation_available", frozenset({True})),
        MembershipPredicate("linear_operator_matrix_available", frozenset({True})),
    )


def budget_predicates() -> tuple[AffineComparisonPredicate, ...]:
    """Return conservative work and memory budget predicates."""
    return (
        AffineComparisonPredicate(
            {"work_budget_fmas": 1.0, "assembly_cost_fmas": -1.0},
            ">=",
            0.0,
        ),
        AffineComparisonPredicate(
            {"work_budget_fmas": 1.0, "matvec_cost_fmas": -1.0},
            ">=",
            0.0,
        ),
        AffineComparisonPredicate(
            {
                "memory_budget_bytes": 1.0,
                "linear_operator_memory_bytes": -1.0,
            },
            ">=",
            0.0,
        ),
    )


__all__: list[str] = []
