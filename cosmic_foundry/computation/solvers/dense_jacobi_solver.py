"""DenseJacobiSolver: matrix-free Jacobi iteration."""

from __future__ import annotations

from typing import Any, NamedTuple

from cosmic_foundry.computation import tensor as tensor_module
from cosmic_foundry.computation.algorithm_capabilities import ComparisonPredicate
from cosmic_foundry.computation.solvers.coverage import (
    CONDITION_LIMIT,
    LINEARITY_TOLERANCE,
)
from cosmic_foundry.computation.solvers.iterative_solver import (
    StationaryIterationSolver,
)
from cosmic_foundry.computation.solvers.linear_solver import LinearOperator
from cosmic_foundry.computation.tensor import Tensor, where


class _JacobiState(NamedTuple):
    u: Tensor
    r: Tensor  # cached residual b − op.apply(u)
    b: Tensor
    diag: Tensor
    omega: Tensor  # 0-d scalar Tensor
    iteration: Tensor  # 0-d int Tensor


class DenseJacobiSolver(StationaryIterationSolver):
    """Jacobi iterative solver for A u = b.

    The damped fixed-point iteration u^{k+1} = u^k + ω D⁻¹(b − Au^k) is a
    contraction when ρ(I − ω D⁻¹A) < 1.  The relaxation factor ω is derived
    automatically from the Gershgorin bound on λ_max(D⁻¹A):

        G = max_i Σ_j |A_{ij}/A_{ii}|   (Gershgorin bound, includes j = i term)
        ω = min(2/G, 1)

    G is an upper bound on λ_max(D⁻¹A) by the Gershgorin circle theorem;
    ω = 2/G guarantees ρ(I − ω D⁻¹A) < 1 whenever λ_max < G.  For
    DiffusiveFlux(2) the interior stencil has G = 2, giving ω = 1 (standard
    Jacobi, the optimal choice).  For DiffusiveFlux(4) the wider stencil
    violates diagonal dominance (G = 32/15 > 2), so standard Jacobi diverges
    and ω = 15/16 is applied automatically.

    Each step uses op.apply() for the residual; no matrix is stored.

    Parameters
    ----------
    tol:
        Convergence tolerance on the squared Euclidean residual ‖b − Au^k‖₂².
    max_iter:
        Hard cap on Jacobi iterations.
    """

    def __init__(self, tol: float = 1e-10, max_iter: int = 100_000) -> None:
        self._tol = tol
        self._max_iter = max_iter

    def init_state(self, op: LinearOperator, b: Tensor) -> _JacobiState:
        backend = b.backend
        n = b.shape[0]
        diag = op.diagonal(backend)
        abs_row_sums = op.row_abs_sums(backend)

        # Gershgorin bound: G = max_i (Σ_j |A_{ij}|) / |A_{ii}|
        lambda_max: Tensor = tensor_module.max(abs_row_sums / tensor_module.abs(diag))
        two_over_lm: Tensor = 2.0 / lambda_max
        omega: Tensor = where(two_over_lm > 1.0, 1.0, two_over_lm)

        u: Tensor = Tensor.zeros(n, backend=backend)
        r: Tensor = b - op.apply(u)
        iteration: Tensor = Tensor(0, backend=backend)
        return _JacobiState(u, r, b, diag, omega, iteration)

    def step(self, op: LinearOperator, state: Any) -> _JacobiState:
        s: _JacobiState = state
        u = s.u + s.omega * (s.r / s.diag)
        r = s.b - op.apply(u)
        return _JacobiState(u, r, s.b, s.diag, s.omega, s.iteration + 1)

    def converged(self, state: Any) -> Tensor:
        s: _JacobiState = state
        return (s.iteration >= self._max_iter) | ((s.r @ s.r) < self._tol**2)

    def extract(self, state: Any) -> Tensor:
        s: _JacobiState = state
        return s.u

    linear_solver_coverage = (
        ComparisonPredicate("diagonal_nonzero_margin", ">", 0.0),
        ComparisonPredicate("diagonal_dominance_margin", ">", 0.0),
        ComparisonPredicate("singular_value_lower_bound", ">", 0.0),
        ComparisonPredicate("condition_estimate", "<=", CONDITION_LIMIT),
        ComparisonPredicate("rhs_consistency_defect", "<=", LINEARITY_TOLERANCE),
    )


__all__ = ["DenseJacobiSolver"]
