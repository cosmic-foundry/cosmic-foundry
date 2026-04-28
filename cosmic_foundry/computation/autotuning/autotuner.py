"""Autotuner: selects the fastest (LinearSolver, Backend) for a ProblemDescriptor."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import NamedTuple

from cosmic_foundry.computation.autotuning.benchmarker import (
    Benchmarker,
    BenchmarkResult,
    fit_log_log,
)
from cosmic_foundry.computation.autotuning.problem_descriptor import ProblemDescriptor
from cosmic_foundry.computation.backends import Backend
from cosmic_foundry.computation.solvers.linear_solver import (
    LinearOperator,
    LinearSolver,
)
from cosmic_foundry.computation.tensor import Tensor


class SelectionResult(NamedTuple):
    """The winning (solver, backend) pair and its predicted cost.

    predicted_cost is in the same units as alpha · N^exponent — a wall-time
    estimate in seconds.  Use it only to compare configurations against
    each other, not as an absolute time bound.
    """

    solver: LinearSolver
    backend: Backend
    predicted_cost: float


class Autotuner:
    """Selects the fastest (LinearSolver, Backend) pair for a given problem.

    calibrate() fits an empirical cost model T ≈ α · N^p for each
    (solver, backend) pair by geometric probing, then prunes dominated pairs
    before running a final high-quality timing at the target N.

      Probe phase (per pair):
        Time at N = 2, 4, 8, … doubling up to descriptor.n // 2.  Detect
        the overhead floor by watching for T(2N)/T(N) ≥ time_ratio_threshold
        (default 2.0): below the floor, T is roughly flat (overhead dominates);
        above it, T grows with N.  Collect (N, T) pairs from the compute-
        dominated regime and fit α and p via log-log linear regression.  If
        fewer than two stable points are found, fall back to a two-point fit
        at descriptor.n // 2 and descriptor.n.

      Prune phase:
        Project each pair's fitted α · descriptor.n^p.  Drop pairs whose
        projected cost exceeds prune_threshold × the best projected cost.

      Final calibration phase (survivors only):
        Time once at descriptor.n with the full warmup + multi-trial protocol.
        Compute α_final = T / descriptor.n^p_fitted, where p_fitted comes from
        the probe phase.  This pins the scale of the cost model to the actual
        target size while keeping the empirically fitted shape (exponent).

    select() evaluates α · N^p for each calibrated pair and returns the
    cheapest.  O(1) per call once calibrated.

    Parameters
    ----------
    solvers:
        Solver instances to consider.  Instances rather than classes because
        solvers carry configuration (tol, max_iter for Jacobi, rcond for SVD).
    backends:
        Backends to benchmark each solver against.
    operator_factory:
        Callable that produces a (LinearOperator, Tensor) pair for a given
        problem size n and backend.  Called at each probe size during
        calibrate(); the caller is responsible for constructing operators
        representative of the target problem class.
    benchmarker:
        Benchmarker instance; defaults to Benchmarker() if not supplied.
    prune_threshold:
        Drop screened pairs whose projected cost exceeds this multiple of the
        best projected cost.  Default 10.0.
    time_ratio_threshold:
        T(2N)/T(N) must reach this value before a point is considered above
        the overhead floor.  Default 2.0: any ratio below this suggests the
        measured time is still flat (overhead dominated).
    """

    def __init__(
        self,
        solvers: Sequence[LinearSolver],
        backends: Sequence[Backend],
        operator_factory: Callable[[int, Backend], tuple[LinearOperator, Tensor]],
        benchmarker: Benchmarker | None = None,
        prune_threshold: float = 10.0,
        time_ratio_threshold: float = 2.0,
    ) -> None:
        self._solvers = list(solvers)
        self._backends = list(backends)
        self._operator_factory = operator_factory
        self._benchmarker = benchmarker if benchmarker is not None else Benchmarker()
        self._prune_threshold = prune_threshold
        self._time_ratio_threshold = time_ratio_threshold
        self._results: list[BenchmarkResult] = []

    def calibrate(self, descriptor: ProblemDescriptor) -> None:
        """Fit cost models for all pairs; prune dominated ones; store final results."""
        all_pairs = [
            (solver, backend) for solver in self._solvers for backend in self._backends
        ]

        # Probe phase: fit (alpha, exponent) per pair from the probe sequence.
        # Fall back to a two-point fit when the probe range is too small.
        probe_fits: list[tuple[LinearSolver, Backend, BenchmarkResult]] = []
        for solver, backend in all_pairs:
            fit = self._fit_cost_model(solver, backend, descriptor)
            if fit is None:
                fit = self._fallback_fit(solver, backend, descriptor)
            probe_fits.append((solver, backend, fit))

        # Prune phase.
        best_projected = min(
            fit.alpha * descriptor.n**fit.exponent for _, _, fit in probe_fits
        )
        survivors = [
            (solver, backend, fit)
            for solver, backend, fit in probe_fits
            if fit.alpha * descriptor.n**fit.exponent
            <= self._prune_threshold * best_projected
        ]

        # Final calibration: one high-quality timing at descriptor.n per survivor.
        # alpha is re-pinned to this measurement; exponent is kept from the probe fit.
        self._results = []
        for solver, backend, probe_fit in survivors:
            op, b = self._operator_factory(descriptor.n, backend)
            t = self._benchmarker.time_solve(solver, op, b)
            alpha = t / descriptor.n**probe_fit.exponent
            self._results.append(
                BenchmarkResult(solver, backend, alpha, probe_fit.exponent)
            )

    def select(self, descriptor: ProblemDescriptor) -> SelectionResult:
        """Return the (solver, backend) pair with minimum predicted cost.

        Predicted cost = α · N^p where α and p are empirically fitted and
        N = descriptor.n.  The descriptor.n used here need not match the
        calibration n — the power-law model extrapolates to any N.

        Raises
        ------
        RuntimeError
            If calibrate() has not been called.
        """
        if not self._results:
            raise RuntimeError("Autotuner.calibrate() must be called before select().")
        best = min(
            self._results,
            key=lambda r: r.alpha * descriptor.n**r.exponent,
        )
        predicted = best.alpha * descriptor.n**best.exponent
        return SelectionResult(best.solver, best.backend, predicted)

    @property
    def results(self) -> list[BenchmarkResult]:
        """Final calibration results for surviving pairs, in iteration order."""
        return list(self._results)

    def _fit_cost_model(
        self,
        solver: LinearSolver,
        backend: Backend,
        descriptor: ProblemDescriptor,
    ) -> BenchmarkResult | None:
        """Probe at N=2,4,…,descriptor.n//2; fit (alpha, exponent) by log-log fit.

        Detects the overhead floor by watching for T(2N)/T(N) >= time_ratio_threshold.
        Below the floor T is flat (overhead dominated); above it T grows as N^p.
        Collects (N, T) pairs from the stable regime and fits by least squares.
        Returns None when fewer than two stable points are collected.
        """
        op, b = self._operator_factory(2, backend)
        prev_t = self._benchmarker.time_solve(solver, op, b)
        stable: list[tuple[int, float]] = []
        found_floor = False

        n = 4
        while n <= descriptor.n // 2:
            op, b = self._operator_factory(n, backend)
            t = self._benchmarker.time_solve(solver, op, b)
            if not found_floor and t / prev_t >= self._time_ratio_threshold:
                found_floor = True
            if found_floor:
                stable.append((n, t))
            prev_t = t
            n *= 2

        if len(stable) < 2:
            return None
        return _make_result(solver, backend, stable)

    def _fallback_fit(
        self,
        solver: LinearSolver,
        backend: Backend,
        descriptor: ProblemDescriptor,
    ) -> BenchmarkResult:
        """Two-point fit at descriptor.n//2 and descriptor.n.

        Used when the probe range is too small to collect two stable points
        (e.g. very small descriptor.n or a backend with a very high overhead floor).
        """
        n1 = max(2, descriptor.n // 2)
        n2 = descriptor.n
        op1, b1 = self._operator_factory(n1, backend)
        t1 = self._benchmarker.time_solve(solver, op1, b1)
        op2, b2 = self._operator_factory(n2, backend)
        t2 = self._benchmarker.time_solve(solver, op2, b2)
        return _make_result(solver, backend, [(n1, t1), (n2, t2)])


def _make_result(
    solver: LinearSolver,
    backend: Backend,
    points: list[tuple[int, float]],
) -> BenchmarkResult:
    """Wrap fit_log_log in a BenchmarkResult for the given solver and backend."""
    alpha, exponent = fit_log_log(points)
    return BenchmarkResult(solver, backend, alpha, exponent)


__all__ = ["Autotuner", "SelectionResult"]
