"""DenseLUSolver: DirectSolver backed by LUFactorization."""

from __future__ import annotations

from cosmic_foundry.computation.algorithm_capabilities import (
    ComparisonPredicate,
    LinearSolverField,
)
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
        ComparisonPredicate(LinearSolverField.DIAGONAL_DOMINANCE_MARGIN, "<=", 0.0),
        ComparisonPredicate(LinearSolverField.SINGULAR_VALUE_LOWER_BOUND, ">", 0.0),
        ComparisonPredicate(
            LinearSolverField.CONDITION_ESTIMATE, "<=", CONDITION_LIMIT
        ),
        ComparisonPredicate(
            LinearSolverField.RHS_CONSISTENCY_DEFECT,
            "<=",
            LINEARITY_TOLERANCE,
        ),
    )


__all__ = ["DenseLUSolver"]
