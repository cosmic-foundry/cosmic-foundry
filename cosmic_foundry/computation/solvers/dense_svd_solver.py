"""DenseSVDSolver: DirectSolver backed by SVDFactorization."""

from __future__ import annotations

from cosmic_foundry.computation.algorithm_capabilities import (
    ComparisonPredicate,
    LinearSolverField,
)
from cosmic_foundry.computation.decompositions.svd_factorization import SVDFactorization
from cosmic_foundry.computation.solvers.coverage import (
    LINEARITY_TOLERANCE,
)
from cosmic_foundry.computation.solvers.direct_solver import DirectSolver


class DenseSVDSolver(DirectSolver):
    """Direct solver for A u = b using thin SVD and the Moore-Penrose pseudoinverse.

    Decomposes A = U Σ Vᵀ and solves via u† = V Σ⁻¹ Uᵀ b, zeroing singular
    values σᵢ < rcond · σ₀.  For full-rank square systems this recovers the
    exact solution; for rank-deficient or overdetermined systems it returns
    the minimum-norm least-squares solution.

    In plain terms: SVD rotates the problem into a coordinate system where A
    acts as a diagonal scaling.  The solve divides by each singular value, then
    rotates back.  Near-zero singular values signal near-null-space directions;
    dividing by them would amplify noise, so they are zeroed out via rcond.

    Prefer DenseLUSolver for full-rank square systems — same O(N³) cost with
    a smaller constant.  Use DenseSVDSolver when A may be rank-deficient (e.g.
    periodic advection), when a minimum-norm solution is required, or when the
    condition number or null-space structure is needed alongside the solution.

    Parameters
    ----------
    rcond:
        Singular values below rcond · σ₀ are treated as zero.
    """

    def __init__(self, rcond: float = 1e-10) -> None:
        super().__init__(SVDFactorization(rcond))

    linear_solver_coverage = (
        ComparisonPredicate(LinearSolverField.NULLITY_ESTIMATE, ">", 0),
        ComparisonPredicate(LinearSolverField.SINGULAR_VALUE_LOWER_BOUND, "<=", 0.0),
        ComparisonPredicate(
            LinearSolverField.RHS_CONSISTENCY_DEFECT,
            "<=",
            LINEARITY_TOLERANCE,
        ),
    )


__all__ = ["DenseSVDSolver"]
