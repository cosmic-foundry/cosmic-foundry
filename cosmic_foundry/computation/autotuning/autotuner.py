"""Autotuner: selects the fastest (LinearSolver, Backend) for a ProblemDescriptor."""

from __future__ import annotations

import math
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

    calibrate() runs a two-phase measurement to avoid spending time on pairs
    that are clearly dominated:

      Screen phase:
        Each (solver, backend) pair is benchmarked at a screen N chosen so
        that N^p >= backend.min_ops.  This keeps the measured time above the
        device's dispatch-overhead noise floor (1 M ops for GPU/TPU, 1 K ops
        for CPU), giving a reliable enough α to compare pairs cheaply.
        Pairs whose min_ops floor would require a screen N >= descriptor.n
        skip the screen entirely and go straight to full calibration.

      Prune phase:
        Any screened pair whose projected cost at descriptor.n exceeds
        prune_threshold × the best screened projected cost is dropped.

      Full calibration phase:
        The surviving pairs are benchmarked at the full descriptor.n with
        the complete warmup + multi-trial protocol.  These results are stored
        and used by select().

    select() applies T = α · Nᵖ to the calibrated results and returns the
    pair with the lowest predicted cost.  O(1) per call once calibrated.

    Parameters
    ----------
    solvers:
        Solver instances to consider.  Instances rather than classes because
        solvers carry configuration (tol, max_iter for Jacobi, rcond for SVD).
    backends:
        Backends to benchmark each solver against.
    benchmarker:
        Benchmarker instance; defaults to Benchmarker() if not supplied.
    prune_threshold:
        A screened pair is dropped if its projected cost at descriptor.n
        exceeds prune_threshold times the best screened projected cost.
        Default 10.0: keep everything within an order of magnitude of the
        current leader.
    """

    def __init__(
        self,
        solvers: Sequence[LinearSolver],
        backends: Sequence[Backend],
        benchmarker: Benchmarker | None = None,
        prune_threshold: float = 10.0,
    ) -> None:
        self._solvers = list(solvers)
        self._backends = list(backends)
        self._benchmarker = benchmarker if benchmarker is not None else Benchmarker()
        self._prune_threshold = prune_threshold
        self._results: list[BenchmarkResult] = []

    def calibrate(self, descriptor: ProblemDescriptor) -> None:
        """Benchmark (solver, backend) pairs at descriptor.n; store α values.

        Runs the screen-then-prune protocol described in the class docstring
        before committing to full-cost calibration on survivors.
        """
        all_pairs = [
            (solver, backend) for solver in self._solvers for backend in self._backends
        ]

        # Screen phase: cheap measurement at the ops-floor N for each pair.
        screened: list[tuple[LinearSolver, Backend, float]] = []
        unscreened: list[tuple[LinearSolver, Backend]] = []

        for solver, backend in all_pairs:
            screen_n = math.ceil(backend.min_ops ** (1.0 / solver.cost_exponent))
            if screen_n >= descriptor.n:
                # Problem is too small for the screen to be cheaper than full
                # calibration — go direct.
                unscreened.append((solver, backend))
            else:
                screen_descriptor = ProblemDescriptor(
                    n=screen_n,
                    g=descriptor.g,
                    r=min(descriptor.r, screen_n),
                    tol=descriptor.tol,
                    spectral_radius=descriptor.spectral_radius,
                )
                result = self._benchmarker.measure(solver, backend, screen_descriptor)
                projected = result.alpha * descriptor.n**solver.cost_exponent
                screened.append((solver, backend, projected))

        # Prune phase: discard screened pairs that are clearly dominated.
        if screened:
            best_projected = min(cost for _, _, cost in screened)
            survivors: list[tuple[LinearSolver, Backend]] = [
                (s, b)
                for s, b, cost in screened
                if cost <= self._prune_threshold * best_projected
            ]
        else:
            survivors = []
        survivors.extend(unscreened)

        # Full calibration on survivors only.
        self._results = [
            self._benchmarker.measure(solver, backend, descriptor)
            for solver, backend in survivors
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
        """Calibration results for surviving pairs, in iteration order."""
        return list(self._results)


__all__ = ["Autotuner", "SelectionResult"]
