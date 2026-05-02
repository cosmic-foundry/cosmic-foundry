"""Shared builders for linear-solver capability declarations."""

from __future__ import annotations

from cosmic_foundry.computation.algorithm_capabilities import (
    AffineComparisonPredicate,
    AlgorithmCapability,
    AlgorithmStructureContract,
    ComparisonPredicate,
    CoveragePatch,
    EvidencePredicate,
    MembershipPredicate,
    StructuredPredicate,
)

LinearSolverCapability = AlgorithmCapability

LINEARITY_TOLERANCE = 1.0e-12
CONDITION_LIMIT = 1.0e8


def contract(
    *,
    requires: tuple[str, ...],
    provides: tuple[str, ...],
) -> AlgorithmStructureContract:
    """Return a linear-solver structure contract."""
    return AlgorithmStructureContract(frozenset(requires), frozenset(provides))


def capability(
    owner: type,
    structure_contract: AlgorithmStructureContract,
    *,
    priority: int | None = None,
    coverage_predicates: tuple[StructuredPredicate, ...] = (),
    coverage_priority: int | None = None,
) -> LinearSolverCapability:
    """Return a capability whose category is inferred from solver inheritance."""
    coverage_patches: tuple[CoveragePatch, ...] = ()
    if coverage_predicates:
        coverage_patches = (
            CoveragePatch(
                owner.__name__,
                owner.__name__,
                "owned",
                coverage_predicates,
                priority=coverage_priority,
            ),
        )
    return LinearSolverCapability(
        owner.__name__,
        owner.__name__,
        category_for(owner),
        structure_contract,
        priority=priority,
        coverage_patches=coverage_patches,
    )


def category_for(owner: type) -> str:
    """Infer the linear-solver category from the implementation class MRO."""
    mro_names = {cls.__name__ for cls in owner.__mro__}
    if "DirectSolver" in mro_names:
        return "direct_solver"
    if "IterativeSolver" in mro_names:
        return "iterative_solver"
    raise ValueError(f"cannot infer linear-solver category for {owner.__name__}")


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


def selector_rejection_patches() -> tuple[CoveragePatch, ...]:
    """Return selector-level rejection regions not owned by implementations."""
    return (
        CoveragePatch(
            "linear_solver_work_budget_below_operator_cost",
            "linear_solver_selector",
            "rejected",
            (
                ComparisonPredicate(
                    "map_linearity_defect",
                    "<=",
                    LINEARITY_TOLERANCE,
                ),
                AffineComparisonPredicate(
                    {"work_budget_fmas": 1.0, "matvec_cost_fmas": -1.0},
                    "<",
                    0.0,
                ),
            ),
        ),
        CoveragePatch(
            "linear_solver_memory_budget_below_operator_storage",
            "linear_solver_selector",
            "rejected",
            (
                ComparisonPredicate(
                    "map_linearity_defect",
                    "<=",
                    LINEARITY_TOLERANCE,
                ),
                AffineComparisonPredicate(
                    {
                        "memory_budget_bytes": 1.0,
                        "linear_operator_memory_bytes": -1.0,
                    },
                    "<",
                    0.0,
                ),
            ),
        ),
        CoveragePatch(
            "linear_solver_unknown_condition",
            "linear_solver_selector",
            "rejected",
            (
                ComparisonPredicate(
                    "map_linearity_defect",
                    "<=",
                    LINEARITY_TOLERANCE,
                ),
                EvidencePredicate("condition_estimate", frozenset({"unavailable"})),
            ),
        ),
    )
