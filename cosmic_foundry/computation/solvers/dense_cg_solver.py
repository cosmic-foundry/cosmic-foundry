"""DenseCGSolver: matrix-free Conjugate Gradient iteration."""

from __future__ import annotations

from typing import Any, NamedTuple

from cosmic_foundry.computation import tensor
from cosmic_foundry.computation.algorithm_capabilities import (
    ComparisonPredicate,
    MembershipPredicate,
)
from cosmic_foundry.computation.solvers.coverage import (
    CONDITION_LIMIT,
    LINEARITY_TOLERANCE,
    budget_predicates,
    linear_system_predicates,
)
from cosmic_foundry.computation.solvers.iterative_solver import KrylovSolver
from cosmic_foundry.computation.solvers.linear_solver import LinearOperator
from cosmic_foundry.computation.tensor import Tensor


class _CGState(NamedTuple):
    u: Tensor
    r: Tensor  # residual b − op.apply(u)
    p: Tensor  # search direction
    rho: Tensor  # 0-d scalar: r @ r
    b: Tensor
    iteration: Tensor  # 0-d int Tensor


class DenseCGSolver(KrylovSolver):
    """Conjugate Gradient solver for A u = b; A must be symmetric positive definite.

    CG minimizes ‖u − u*‖_A² over successive Krylov subspaces
    K_k = span{r₀, Ar₀, …, A^{k−1}r₀}, generating A-conjugate search
    directions {p_k}.  For an N × N SPD matrix with condition number κ(A),
    CG converges to ε-accuracy in O(√κ · log(2/ε)) iterations; each iteration
    costs one matvec via op.apply (O(nnz)).

    Each iteration:

        q     = op.apply(p)
        α     = ρ / (p · q)        where ρ = r · r
        u     = u + α p
        r     = r − α q
        ρ_new = r · r
        β     = ρ_new / ρ
        p     = r + β p
        ρ     = ρ_new

    Do NOT use for non-symmetric or indefinite matrices.

    Parameters
    ----------
    tol:
        Convergence tolerance: iteration stops when ρ = ‖r‖₂² < tol².
    max_iter:
        Hard cap on CG iterations.
    """

    def __init__(self, tol: float = 1e-10, max_iter: int = 100_000) -> None:
        self._tol = tol
        self._max_iter = max_iter

    def init_state(self, op: LinearOperator, b: Tensor) -> _CGState:
        u = Tensor.zeros(b.shape[0], backend=b.backend)
        r = tensor.copy(b)  # r₀ = b − A·0 = b
        p = tensor.copy(r)
        rho = r @ r
        iteration: Tensor = Tensor(0, backend=b.backend)
        return _CGState(u, r, p, rho, b, iteration)

    def step(self, op: LinearOperator, state: Any) -> _CGState:
        s: _CGState = state
        q = op.apply(s.p)
        alpha = s.rho / (s.p @ q)
        u = s.u + alpha * s.p
        r = s.r - alpha * q
        rho_new = r @ r
        beta = rho_new / s.rho
        p = r + beta * s.p
        return _CGState(u, r, p, rho_new, s.b, s.iteration + 1)

    def converged(self, state: Any) -> Tensor:
        s: _CGState = state
        return (s.iteration >= self._max_iter) | (s.rho < self._tol**2)

    def extract(self, state: Any) -> Tensor:
        s: _CGState = state
        return s.u

    linear_solver_coverage = (
        linear_system_predicates()
        + budget_predicates()
        + (
            MembershipPredicate("operator_application_available", frozenset({True})),
            ComparisonPredicate("symmetry_defect", "<=", LINEARITY_TOLERANCE),
            ComparisonPredicate("coercivity_lower_bound", ">", 0.0),
            ComparisonPredicate("singular_value_lower_bound", ">", 0.0),
            ComparisonPredicate("condition_estimate", "<=", CONDITION_LIMIT),
            ComparisonPredicate("rhs_consistency_defect", "<=", LINEARITY_TOLERANCE),
        )
    )
    linear_solver_coverage_priority = 10


__all__ = ["DenseCGSolver"]
