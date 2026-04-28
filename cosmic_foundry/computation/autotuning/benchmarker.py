"""Benchmarker: times a single solver against a pre-built operator."""

from __future__ import annotations

import math
import time
from typing import NamedTuple

from cosmic_foundry.computation.backends import Backend
from cosmic_foundry.computation.solvers.linear_solver import (
    LinearOperator,
    LinearSolver,
)
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
    """Times a solver on a caller-supplied operator and right-hand side.

    Runs n_warmup solves to let JIT compilers amortize compilation, then
    takes the minimum over n_trials timed solves.  Taking the minimum
    rather than the mean eliminates OS scheduling noise while still
    catching algorithmic slowdowns.

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
        op: LinearOperator,
        b: Tensor,
    ) -> float:
        """Return the minimum wall time in seconds for one solve."""
        for _ in range(self._n_warmup):
            result = solver.solve(op, b)
            result.sync()

        best = float("inf")
        for _ in range(self._n_trials):
            t0 = time.perf_counter()
            result = solver.solve(op, b)
            result.sync()
            best = min(best, time.perf_counter() - t0)

        return best


def fit_log_log(points: list[tuple[int, float]]) -> tuple[float, float]:
    """Fit T = alpha * N^exponent to (N, T) pairs by log-log linear regression.

    Parameters
    ----------
    points:
        List of (N, T) pairs where N is problem size and T is wall time in
        seconds.  Must contain at least two distinct N values.

    Returns
    -------
    (alpha, exponent):
        Coefficients of the fitted power law T = alpha * N^exponent.
    """
    xs = [math.log(n) for n, _ in points]
    ys = [math.log(t) for _, t in points]
    k = len(xs)
    mx = sum(xs) / k
    my = sum(ys) / k
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=False))
    den = sum((x - mx) ** 2 for x in xs)
    exponent = num / den if den else 1.0
    alpha = math.exp(my - exponent * mx)
    return alpha, exponent


__all__ = ["BenchmarkResult", "Benchmarker", "fit_log_log"]
