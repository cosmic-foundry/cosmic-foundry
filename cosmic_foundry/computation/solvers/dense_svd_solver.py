"""DenseSVDSolver: DirectSolver backed by SVDFactorization."""

from __future__ import annotations

from cosmic_foundry.computation.algorithm_capabilities import ComparisonPredicate
from cosmic_foundry.computation.decompositions.svd_factorization import SVDFactorization
from cosmic_foundry.computation.solvers._capability_claims import (
    LINEARITY_TOLERANCE,
    LinearSolverCapability,
    budget_predicates,
    contract,
    dense_matrix_predicates,
    linear_system_predicates,
    owned_patch,
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

    _coverage_patch = owned_patch(
        "dense_svd_rank_deficient_dense",
        "DenseSVDSolver",
        linear_system_predicates()
        + dense_matrix_predicates()
        + budget_predicates()
        + (
            ComparisonPredicate("nullity_estimate", ">", 0),
            ComparisonPredicate("rhs_consistency_defect", "<=", LINEARITY_TOLERANCE),
        ),
        priority=20,
    )

    @classmethod
    def linear_solver_capabilities(cls) -> tuple[LinearSolverCapability, ...]:
        """Return capability declarations owned by this solver implementation."""
        return (
            LinearSolverCapability(
                "dense_svd_direct",
                cls.__name__,
                "direct_solver",
                contract(
                    requires=("dense_operator",),
                    provides=(
                        "solve",
                        "direct",
                        "least_squares",
                        "minimum_norm",
                        "rank_deficient",
                        "factorized_dense",
                    ),
                ),
                coverage_patches=(cls._coverage_patch,),
            ),
        )


__all__ = ["DenseSVDSolver"]
