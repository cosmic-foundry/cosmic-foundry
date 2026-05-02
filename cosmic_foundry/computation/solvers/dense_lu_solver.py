"""DenseLUSolver: DirectSolver backed by LUFactorization."""

from __future__ import annotations

from cosmic_foundry.computation.algorithm_capabilities import ComparisonPredicate
from cosmic_foundry.computation.decompositions.lu_factorization import LUFactorization
from cosmic_foundry.computation.solvers.coverage import (
    CONDITION_LIMIT,
    LINEARITY_TOLERANCE,
)
from cosmic_foundry.computation.solvers.direct_solver import DirectSolver


class DenseLUSolver(DirectSolver):
    """Direct solver for A u = b using LU factorization with partial pivoting.

    Convenience wrapper around DirectSolver(LUFactorization()).
    See LUFactorization for algorithm details.
    """

    def __init__(self) -> None:
        super().__init__(LUFactorization())

    linear_solver_coverage = (
        ComparisonPredicate("diagonal_dominance_margin", "<=", 0.0),
        ComparisonPredicate("singular_value_lower_bound", ">", 0.0),
        ComparisonPredicate("condition_estimate", "<=", CONDITION_LIMIT),
        ComparisonPredicate("rhs_consistency_defect", "<=", LINEARITY_TOLERANCE),
    )


__all__ = ["DenseLUSolver"]
