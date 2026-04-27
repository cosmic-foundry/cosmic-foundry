"""Autotuner: selects the fastest (LinearSolver, Backend) for a ProblemDescriptor."""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

from cosmic_foundry.computation.autotuning.benchmarker import (
    Benchmarker,
    BenchmarkResult,
)
from cosmic_foundry.computation.autotuning.problem_descriptor import ProblemDescriptor
from cosmic_foundry.computation.backends import Backend
from cosmic_foundry.computation.solvers.linear_solver import LinearSolver


class SelectionResult(NamedTuple):
    """The winning (solver, backend) pair and its predicted cost.

    predicted_cost is in the same units as alpha · Nᵖ — a wall-time
    estimate in seconds.  Use it only to compare configurations against
    each other, not as an absolute time bound.
    """

    solver: LinearSolver
    backend: Backend
    predicted_cost: float


class Autotuner:
    """Selects the fastest (LinearSolver, Backend) pair for a given problem.

    Two-phase design separates the expensive measurement from the cheap
    prediction so calibration can be paid once and reused across many
    queries:

      Phase 1 — calibrate(descriptor):
        Runs the Benchmarker on every (solver, backend) pair and stores the
        empirical α coefficients.  Expensive: one full solve per pair.

      Phase 2 — select(descriptor):
        Applies the cost model T = α · Nᵖ to predict solve time for each
        pair at descriptor.n and returns the fastest.  O(1) per query.

    The cost model is purely analytic given the calibrated α values.  The
    solver choice is the one with the minimum predicted cost; no further
    benchmark solves are run during select().

    Parameters
    ----------
    solvers:
        Solver instances to consider.  Instances rather than classes because
        solvers carry configuration (tol, max_iter for Jacobi, rcond for SVD).
    backends:
        Backends to benchmark each solver against.
    benchmarker:
        Benchmarker instance; defaults to Benchmarker() if not supplied.
    """

    def __init__(
        self,
        solvers: Sequence[LinearSolver],
        backends: Sequence[Backend],
        benchmarker: Benchmarker | None = None,
    ) -> None:
        self._solvers = list(solvers)
        self._backends = list(backends)
        self._benchmarker = benchmarker if benchmarker is not None else Benchmarker()
        self._results: list[BenchmarkResult] = []

    def calibrate(self, descriptor: ProblemDescriptor) -> None:
        """Benchmark every (solver, backend) pair at descriptor.n; store α values."""
        self._results = [
            self._benchmarker.measure(solver, backend, descriptor)
            for solver in self._solvers
            for backend in self._backends
        ]

    def select(self, descriptor: ProblemDescriptor) -> SelectionResult:
        """Return the (solver, backend) pair with minimum predicted cost.

        Predicted cost = α · Nᵖ where α is the calibrated coefficient, p is
        solver.cost_exponent, and N = descriptor.n.  The descriptor.n used
        here need not match the calibration n — the model extrapolates to any N.

        Raises
        ------
        RuntimeError
            If calibrate() has not been called.
        """
        if not self._results:
            raise RuntimeError("Autotuner.calibrate() must be called before select().")
        best = min(
            self._results,
            key=lambda r: r.alpha * descriptor.n**r.solver.cost_exponent,
        )
        predicted = best.alpha * descriptor.n**best.solver.cost_exponent
        return SelectionResult(best.solver, best.backend, predicted)

    @property
    def results(self) -> list[BenchmarkResult]:
        """Calibration results, in (solver, backend) iteration order."""
        return list(self._results)


__all__ = ["Autotuner", "SelectionResult"]
