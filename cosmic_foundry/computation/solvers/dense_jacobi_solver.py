"""DenseJacobiSolver: Jacobi iteration on an assembled dense matrix."""

from __future__ import annotations

from typing import Any, NamedTuple

from cosmic_foundry.computation.solvers.iterative_solver import IterativeSolver
from cosmic_foundry.computation.tensor import Tensor, einsum, where


class _JacobiState(NamedTuple):
    u: Tensor
    r: Tensor  # cached residual b - a @ u
    a: Tensor
    b: Tensor
    diag: Tensor
    omega: Tensor  # 0-d scalar Tensor
    iteration: Tensor  # 0-d int Tensor (trace-compatible: no Python int in state)


class DenseJacobiSolver(IterativeSolver):
    """Jacobi iterative solver for A u = b on an N × N dense matrix.

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

    In plain terms: split A = D − (D − A) where D = diag(A).  Each damped
    Jacobi step scales the correction by ω before applying the diagonal inverse.
    With ω derived from the Gershgorin bound the iteration contracts for any
    SPD operator assembled by FVMDiscretization with DirichletBC, regardless of
    stencil width.

    Parameters
    ----------
    tol:
        Convergence tolerance on the squared Euclidean residual ‖b − Au^k‖₂².
    max_iter:
        Hard cap on Jacobi iterations.
    """

    cost_exponent = 4  # O(N^2) iterations × O(N^2) matvec per iteration

    def __init__(self, tol: float = 1e-10, max_iter: int = 100_000) -> None:
        self._tol = tol
        self._max_iter = max_iter

    def init_state(self, a: Tensor, b: Tensor) -> _JacobiState:
        n = a.shape[0]
        diag: Tensor = a.diag()

        # Gershgorin bound on λ_max(D⁻¹A): ω = min(2/G, 1) guarantees contraction.
        # G = max_i Σ_j |A_{ij} / A_{ii}|  (row sums of |D⁻¹A|, including diagonal)
        row_sums: Tensor = einsum("ij->i", a.abs()) / diag.abs()
        lambda_max: Tensor = row_sums.max()  # 0-d Tensor
        two_over_lm: Tensor = 2.0 / lambda_max  # __rtruediv__, 0-d Tensor
        omega: Tensor = where(two_over_lm > 1.0, 1.0, two_over_lm)

        u: Tensor = Tensor.zeros(n, backend=a.backend)
        r: Tensor = b - a @ u
        iteration: Tensor = Tensor(0, backend=a.backend)
        return _JacobiState(u, r, a, b, diag, omega, iteration)

    def step(self, state: Any) -> _JacobiState:
        s: _JacobiState = state
        u = s.u + s.omega * (s.r / s.diag)
        r = s.b - s.a @ u
        return _JacobiState(u, r, s.a, s.b, s.diag, s.omega, s.iteration + 1)

    def converged(self, state: Any) -> Tensor:
        s: _JacobiState = state
        max_iter_reached = s.iteration >= self._max_iter
        residual_small = (s.r @ s.r) < self._tol**2
        return max_iter_reached | residual_small

    def extract(self, state: Any) -> Tensor:
        s: _JacobiState = state
        return s.u


__all__ = ["DenseJacobiSolver"]
