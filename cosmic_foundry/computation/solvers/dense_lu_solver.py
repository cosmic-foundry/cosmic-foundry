"""DenseLUSolver: DirectSolver backed by LUFactorization."""

from __future__ import annotations

from cosmic_foundry.computation.algorithm_capabilities import ComparisonPredicate
from cosmic_foundry.computation.decompositions.lu_factorization import LUFactorization
from cosmic_foundry.computation.solvers._capability_claims import (
    CONDITION_LIMIT,
    LINEARITY_TOLERANCE,
    LinearSolverCapability,
    budget_predicates,
    contract,
    dense_matrix_predicates,
    linear_system_predicates,
    owned_patch,
)
from cosmic_foundry.computation.solvers.direct_solver import DirectSolver


class DenseLUSolver(DirectSolver):
    """Direct solver for A u = b using LU factorization with partial pivoting.

    Convenience wrapper around DirectSolver(LUFactorization()).
    See LUFactorization for algorithm details.
    """

    def __init__(self) -> None:
        super().__init__(LUFactorization())


_COVERAGE_PATCH = owned_patch(
    "dense_lu_square_full_rank_dense",
    "DenseLUSolver",
    linear_system_predicates()
    + dense_matrix_predicates()
    + budget_predicates()
    + (
        ComparisonPredicate("singular_value_lower_bound", ">", 0.0),
        ComparisonPredicate("condition_estimate", "<=", CONDITION_LIMIT),
        ComparisonPredicate("rhs_consistency_defect", "<=", LINEARITY_TOLERANCE),
    ),
    priority=30,
)


def declare_linear_solver_capabilities() -> tuple[LinearSolverCapability, ...]:
    """Return capability declarations owned by this solver implementation."""
    return (
        LinearSolverCapability(
            "dense_lu_direct",
            "DenseLUSolver",
            "direct_solver",
            contract(
                requires=("dense_operator", "square_system", "full_rank"),
                provides=("solve", "direct", "exact", "factorized_dense"),
            ),
            priority=10,
            coverage_patches=(_COVERAGE_PATCH,),
        ),
    )


__all__ = ["DenseLUSolver", "declare_linear_solver_capabilities"]
