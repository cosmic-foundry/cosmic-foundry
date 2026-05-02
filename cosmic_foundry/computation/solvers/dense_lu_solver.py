"""DenseLUSolver: DirectSolver backed by LUFactorization."""

from __future__ import annotations

from cosmic_foundry.computation.algorithm_capabilities import ComparisonPredicate
from cosmic_foundry.computation.decompositions.lu_factorization import LUFactorization
from cosmic_foundry.computation.solvers._capability_claims import (
    CONDITION_LIMIT,
    LINEARITY_TOLERANCE,
    LinearSolverCapability,
    budget_predicates,
    capability,
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

    _coverage_patch = owned_patch(
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

    @classmethod
    def linear_solver_capabilities(cls) -> tuple[LinearSolverCapability, ...]:
        """Return capability declarations owned by this solver implementation."""
        return (
            capability(
                cls,
                "dense_lu_direct",
                contract(
                    requires=("dense_operator", "square_system", "full_rank"),
                    provides=("solve", "direct", "exact", "factorized_dense"),
                ),
                priority=10,
                coverage_patches=(cls._coverage_patch,),
            ),
        )


__all__ = ["DenseLUSolver"]
