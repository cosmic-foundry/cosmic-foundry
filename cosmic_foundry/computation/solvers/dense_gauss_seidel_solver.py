"""DenseGaussSeidelSolver: Gauss-Seidel iteration (assembles matrix once in init)."""

from __future__ import annotations

from typing import Any, NamedTuple

from cosmic_foundry.computation import tensor
from cosmic_foundry.computation.solvers._capability_claims import (
    LinearSolverCapability,
    contract,
)
from cosmic_foundry.computation.solvers.iterative_solver import IterativeSolver
from cosmic_foundry.computation.solvers.linear_solver import LinearOperator
from cosmic_foundry.computation.tensor import Tensor


def _gs_sweep_body(i: Any, state: tuple) -> tuple:
    u, a, b, diag = state
    correction = (b[i] - a[i] @ u) / diag[i]
    return (u.set(i, u[i] + correction), a, b, diag)


class _GSState(NamedTuple):
    u: Tensor
    r: Tensor  # cached residual b - a @ u
    a: Tensor  # assembled matrix (built once in init_state)
    b: Tensor
    diag: Tensor
    iteration: Tensor  # 0-d int Tensor


class DenseGaussSeidelSolver(IterativeSolver):
    """Gauss-Seidel iterative solver for A u = b.

    Each sweep updates every component sequentially:

        u_i ← (b_i − Σ_{j<i} A_{ij} u_j^{new} − Σ_{j>i} A_{ij} u_j^{old}) / A_{ii}

    using already-updated values u_1…u_{i-1} from the current sweep.  This
    is equivalent to splitting A = L + D + U (strict lower, diagonal, strict
    upper) and solving (L + D) u^{k+1} = b − U u^k at each step.  The
    iteration converges whenever A is SPD or strictly diagonally dominant.

    The sequential sweep requires explicit access to individual matrix entries
    A[i,j], so the stiffness matrix is assembled once from op.apply() during
    init_state and stored in the solver state.  Each Gauss-Seidel step uses
    the stored matrix — op is not called again after initialization.

    The sweep is implemented via backend.fori_loop so it compiles to a single
    XLA kernel on JaxBackend.  The sequential update is inherent: the i-th
    update depends on u[0..i-1] from the current sweep, so the inner loop
    cannot be parallelized across i.  For GPU workloads prefer DenseCGSolver.

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

    def _assemble(self, op: LinearOperator, b: Tensor) -> Tensor:
        n = b.shape[0]
        backend = b.backend
        columns: list[list[float]] = []
        for j in range(n):
            e_j = Tensor.zeros(n, backend=backend)
            e_j = e_j.set(j, Tensor(1.0, backend=backend))
            columns.append(backend.flatten(op.apply(e_j)._value))
        rows = [[columns[j][i] for j in range(n)] for i in range(n)]
        return Tensor(rows, backend=backend)

    def init_state(self, op: LinearOperator, b: Tensor) -> _GSState:
        a = self._assemble(op, b)
        n = b.shape[0]
        diag = tensor.diag(a)
        u = Tensor.zeros(n, backend=b.backend)
        r = b - a @ u
        iteration: Tensor = Tensor(0, backend=b.backend)
        return _GSState(u, r, a, b, diag, iteration)

    def step(self, op: LinearOperator, state: Any) -> _GSState:
        s: _GSState = state
        n = s.a.shape[0]
        u_new, *_ = s.a.backend.fori_loop(n, _gs_sweep_body, (s.u, s.a, s.b, s.diag))
        r_new = s.b - s.a @ u_new
        return _GSState(u_new, r_new, s.a, s.b, s.diag, s.iteration + 1)

    def converged(self, state: Any) -> Tensor:
        s: _GSState = state
        return (s.iteration >= self._max_iter) | ((s.r @ s.r) < self._tol**2)

    def extract(self, state: Any) -> Tensor:
        s: _GSState = state
        return s.u

    @classmethod
    def linear_solver_capabilities(cls) -> tuple[LinearSolverCapability, ...]:
        """Return capability declarations owned by this solver implementation."""
        return (
            LinearSolverCapability(
                "dense_gauss_seidel_iteration",
                cls.__name__,
                "iterative_solver",
                contract(
                    requires=("linear_operator", "square_system"),
                    provides=("solve", "iterative", "stationary", "assembled_matrix"),
                ),
            ),
        )


__all__ = ["DenseGaussSeidelSolver"]
