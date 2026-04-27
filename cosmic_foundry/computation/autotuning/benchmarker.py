"""Benchmarker: times a single solver/backend/size triple."""

from __future__ import annotations

import time
from typing import NamedTuple

from cosmic_foundry.computation.autotuning.problem_descriptor import ProblemDescriptor
from cosmic_foundry.computation.backends import Backend
from cosmic_foundry.computation.solvers.linear_solver import LinearSolver
from cosmic_foundry.computation.tensor import Tensor


class BenchmarkResult(NamedTuple):
    """Empirical cost-model fit for one (solver, backend) pair.

    alpha and exponent satisfy T_solve(N) ≈ alpha · N^exponent, where
    T_solve is the wall time of a single solve at size N.  Both values are
    fitted empirically by Autotuner from a geometric probe sequence; neither
    is derived from theoretical complexity.
    """

    solver: LinearSolver
    backend: Backend
    alpha: float
    exponent: float


class Benchmarker:
    """Times a solver/backend pair on a synthetic problem of a given size.

    Generates a synthetic SPD matrix representative of the problem class,
    runs n_warmup solves to let JIT compilers amortize compilation, then
    takes the minimum over n_trials timed solves.  Taking the minimum
    rather than the mean eliminates OS scheduling noise while still
    catching algorithmic slowdowns.

    The synthetic matrix is a tridiagonal SPD matrix (diagonal = 2,
    off-diagonals = −1) representative of the Poisson stiffness matrix.
    This gives G ≈ 2 for interior rows regardless of descriptor.g.  The
    g parameter is used by Autotuner for cost-model predictions but not
    for matrix generation; future work can generate descriptor-matched
    matrices for problems with G ≠ 2.

    For rank-deficient problems (descriptor.r < descriptor.n), the last
    (n − r) rows and columns are zeroed, producing a rank-r matrix.

    Parameters
    ----------
    n_warmup:
        Number of warm-up solves before timing begins.  Allows XLA/JAX to
        compile and cache the kernel so timing reflects steady-state cost.
    n_trials:
        Number of timed solves.  The minimum elapsed time is returned.
    """

    def __init__(self, n_warmup: int = 1, n_trials: int = 5) -> None:
        self._n_warmup = n_warmup
        self._n_trials = n_trials

    def time_solve(
        self,
        solver: LinearSolver,
        backend: Backend,
        descriptor: ProblemDescriptor,
    ) -> float:
        """Return the minimum wall time in seconds for one solve at descriptor.n."""
        a = self._make_matrix(descriptor, backend)
        b: Tensor = Tensor([1.0] * descriptor.n, backend=backend)

        for _ in range(self._n_warmup):
            result = solver.solve(a, b)
            result.sync()

        best = float("inf")
        for _ in range(self._n_trials):
            t0 = time.perf_counter()
            result = solver.solve(a, b)
            result.sync()
            best = min(best, time.perf_counter() - t0)

        return best

    @staticmethod
    def _make_matrix(descriptor: ProblemDescriptor, backend: Backend) -> Tensor:
        """Synthetic SPD tridiagonal matrix of size n with numerical rank r.

        Interior rows have G = 2 (diagonal-dominant: matches Poisson stiffness).
        Rows and columns beyond index r − 1 are zeroed to produce rank r.
        """
        n = descriptor.n
        r = min(descriptor.r, n)
        rows = [[0.0] * n for _ in range(n)]
        for i in range(r):
            rows[i][i] = 2.0
            if i > 0:
                rows[i][i - 1] = -1.0
            if i < r - 1:
                rows[i][i + 1] = -1.0
        return Tensor(rows, backend=backend)


__all__ = ["BenchmarkResult", "Benchmarker"]
