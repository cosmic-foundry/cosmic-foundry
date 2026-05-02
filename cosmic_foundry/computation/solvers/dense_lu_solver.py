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
    dense_matrix_predicates,
    linear_system_predicates,
)
from cosmic_foundry.computation.solvers.direct_solver import DirectSolver


class DenseLUSolver(DirectSolver):
    """Direct solver for A u = b using LU factorization with partial pivoting.

    Convenience wrapper around DirectSolver(LUFactorization()).
    See LUFactorization for algorithm details.
    """

    def __init__(self) -> None:
        super().__init__(LUFactorization())

    _coverage_predicates = (
        linear_system_predicates()
        + dense_matrix_predicates()
        + budget_predicates()
        + (
            ComparisonPredicate("singular_value_lower_bound", ">", 0.0),
            ComparisonPredicate("condition_estimate", "<=", CONDITION_LIMIT),
            ComparisonPredicate("rhs_consistency_defect", "<=", LINEARITY_TOLERANCE),
        )
    )

    @classmethod
    def linear_solver_capabilities(cls) -> tuple[LinearSolverCapability, ...]:
        """Return descriptor-space coverage owned by this solver implementation."""
        return (
            capability(
                cls,
                coverage_predicates=cls._coverage_predicates,
                coverage_priority=30,
            ),
        )


__all__ = ["DenseLUSolver"]
