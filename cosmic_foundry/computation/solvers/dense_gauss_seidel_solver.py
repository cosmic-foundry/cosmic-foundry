"""DenseGaussSeidelSolver: Gauss-Seidel iteration on an assembled dense matrix."""

from __future__ import annotations

from typing import Any, NamedTuple

from cosmic_foundry.computation import tensor
from cosmic_foundry.computation.solvers.iterative_solver import IterativeSolver
from cosmic_foundry.computation.tensor import Tensor


def _gs_sweep_body(i: Any, state: tuple) -> tuple:
    u, a, b, diag = state
    # u_i ← u_i + (b_i − A[i]·u) / A[ii]; A[i]·u incorporates already-updated
    # u[0..i-1] from this sweep because u is updated in place via fori_loop.
    correction = (b[i] - a[i] @ u) / diag[i]
    return (u.set(i, u[i] + correction), a, b, diag)


class _GSState(NamedTuple):
    u: Tensor
    r: Tensor  # cached residual b - a @ u
    a: Tensor
    b: Tensor
    diag: Tensor
    iteration: Tensor  # 0-d int Tensor


class DenseGaussSeidelSolver(IterativeSolver):
    """Gauss-Seidel iterative solver for A u = b on an N × N dense matrix.

    Each sweep updates every component sequentially:

        u_i ← (b_i − Σ_{j<i} A_{ij} u_j^{new} − Σ_{j>i} A_{ij} u_j^{old}) / A_{ii}

    using already-updated values u_1…u_{i-1} from the current sweep.  This
    is equivalent to splitting A = L + D + U (strict lower, diagonal, strict
    upper) and solving (L + D) u^{k+1} = b − U u^k at each step.  The
    iteration converges whenever A is SPD or strictly diagonally dominant.

    In plain terms: unlike Jacobi, which updates all components simultaneously
    from the old values, Gauss-Seidel immediately uses each new value as it
    is computed.  This halves the number of sweeps to convergence for SPD
    systems (by the Ostrowski-Reich theorem) compared to Jacobi.

    The sweep is implemented via backend.fori_loop so it compiles to a single
    XLA kernel on JaxBackend.  The sequential update is inherent: the i-th
    update depends on u[0..i-1] from the current sweep, so the inner loop
    cannot be parallelized across i.  For GPU workloads prefer
    DenseCGSolver — it converges in O(√κ) iterations with fully parallel
    matvec steps.

    Parameters
    ----------
    tol:
        Convergence tolerance on the squared Euclidean residual ‖b − Au^k‖₂².
    max_iter:
        Hard cap on Gauss-Seidel sweeps.
    """

    def __init__(self, tol: float = 1e-10, max_iter: int = 100_000) -> None:
        self._tol = tol
        self._max_iter = max_iter

    def init_state(self, a: Tensor, b: Tensor) -> _GSState:
        n = a.shape[0]
        diag = tensor.diag(a)
        u = Tensor.zeros(n, backend=a.backend)
        r = b - a @ u
        iteration: Tensor = Tensor(0, backend=a.backend)
        return _GSState(u, r, a, b, diag, iteration)

    def step(self, state: Any) -> _GSState:
        s: _GSState = state
        n = s.a.shape[0]
        u_new, *_ = s.a.backend.fori_loop(n, _gs_sweep_body, (s.u, s.a, s.b, s.diag))
        r_new = s.b - s.a @ u_new
        return _GSState(u_new, r_new, s.a, s.b, s.diag, s.iteration + 1)

    def converged(self, state: Any) -> Tensor:
        s: _GSState = state
        max_iter_reached = s.iteration >= self._max_iter
        residual_small = (s.r @ s.r) < self._tol**2
        return max_iter_reached | residual_small

    def extract(self, state: Any) -> Tensor:
        s: _GSState = state
        return s.u


__all__ = ["DenseGaussSeidelSolver"]
