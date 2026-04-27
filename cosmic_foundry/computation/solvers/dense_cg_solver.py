"""DenseCGSolver: Conjugate Gradient iteration on an assembled dense matrix."""

from __future__ import annotations

from typing import Any, NamedTuple

from cosmic_foundry.computation import tensor
from cosmic_foundry.computation.solvers.iterative_solver import IterativeSolver
from cosmic_foundry.computation.tensor import Tensor


class _CGState(NamedTuple):
    u: Tensor
    r: Tensor  # residual b - a @ u
    p: Tensor  # search direction
    rho: Tensor  # 0-d scalar: r @ r
    a: Tensor
    b: Tensor
    iteration: Tensor  # 0-d int Tensor


class DenseCGSolver(IterativeSolver):
    """Conjugate Gradient solver for A u = b; A must be symmetric positive definite.

    CG minimizes ‖u − u*‖_A² over successive Krylov subspaces
    K_k = span{r₀, Ar₀, …, A^{k−1}r₀}, generating A-conjugate search
    directions {p_k}.  For an N × N SPD matrix with condition number κ(A),
    CG converges to ε-accuracy in O(√κ · log(2/ε)) iterations; each iteration
    costs one matvec A p (O(N²)).  Total cost O(√κ N²) compares favourably
    to O(κ N²) for Jacobi and Gauss-Seidel.

    Each iteration:

        q     = A p
        α     = ρ / (p · q)        where ρ = r · r
        u     = u + α p
        r     = r − α q
        ρ_new = r · r
        β     = ρ_new / ρ
        p     = r + β p
        ρ     = ρ_new

    In plain terms: each CG step picks a direction p that is A-orthogonal
    (conjugate) to all previous directions, so no previously-made progress is
    undone.  Unlike Jacobi or Gauss-Seidel, which update one component at a
    time, CG moves through the full solution space, reaching the exact answer in
    at most N steps for an N × N system.  In practice convergence is much
    faster: O(√κ) iterations rather than O(κ).

    Every operation (matvec, dot products, vector updates) is fully parallel.
    Under JaxBackend all steps compile to efficient GPU kernels with no
    sequential inner loop, making this the preferred iterative solver for
    large SPD systems on accelerators.

    Do NOT use for non-symmetric or indefinite matrices — CG can diverge.
    For rank-deficient or near-singular systems use DenseSVDSolver.  For
    non-SPD matrices use DenseJacobiSolver, DenseGaussSeidelSolver, or
    DenseLUSolver.

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

    def init_state(self, a: Tensor, b: Tensor) -> _CGState:
        u = Tensor.zeros(b.shape[0], backend=a.backend)
        r = tensor.copy(b)  # r₀ = b − A·0 = b
        p = tensor.copy(r)
        rho = r @ r
        iteration: Tensor = Tensor(0, backend=a.backend)
        return _CGState(u, r, p, rho, a, b, iteration)

    def step(self, state: Any) -> _CGState:
        s: _CGState = state
        q = s.a @ s.p
        alpha = s.rho / (s.p @ q)
        u = s.u + alpha * s.p
        r = s.r - alpha * q
        rho_new = r @ r
        beta = rho_new / s.rho
        p = r + beta * s.p
        return _CGState(u, r, p, rho_new, s.a, s.b, s.iteration + 1)

    def converged(self, state: Any) -> Tensor:
        s: _CGState = state
        max_iter_reached = s.iteration >= self._max_iter
        residual_small = s.rho < self._tol**2
        return max_iter_reached | residual_small

    def extract(self, state: Any) -> Tensor:
        s: _CGState = state
        return s.u


__all__ = ["DenseCGSolver"]
