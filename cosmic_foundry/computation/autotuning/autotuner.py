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

    calibrate() runs a three-phase measurement to avoid spending time on
    pairs that are clearly dominated:

      Overhead-detection phase:
        For each (solver, backend) pair, probe at N = 2, 4, 8, … doubling
        until consecutive measured α values agree within alpha_stability.
        That convergence signals that compute cost dominates dispatch
        overhead at that N, giving a reliable baseline α without any
        hardcoded device constants.  Pairs for which no stable N is found
        below descriptor.n skip screening and go directly to full calibration.

      Prune phase:
        Project each stable screen α to descriptor.n via α · Nᵖ.  Drop any
        pair whose projected cost exceeds prune_threshold × the best
        projected cost.

      Full calibration phase:
        Run the complete warmup + multi-trial benchmark at descriptor.n for
        surviving pairs only.  These results are stored and used by select().

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
    alpha_stability:
        Consecutive screen probes are considered stable when
        alpha_prev / alpha_curr < alpha_stability.  Default 1.5: accept
        once alpha has stopped inflating by more than 50% per doubling.
    """

    def __init__(
        self,
        solvers: Sequence[LinearSolver],
        backends: Sequence[Backend],
        benchmarker: Benchmarker | None = None,
        prune_threshold: float = 10.0,
        alpha_stability: float = 1.5,
    ) -> None:
        self._solvers = list(solvers)
        self._backends = list(backends)
        self._benchmarker = benchmarker if benchmarker is not None else Benchmarker()
        self._prune_threshold = prune_threshold
        self._alpha_stability = alpha_stability
        self._results: list[BenchmarkResult] = []

    def calibrate(self, descriptor: ProblemDescriptor) -> None:
        """Benchmark (solver, backend) pairs at descriptor.n; store α values.

        Runs the overhead-detection → prune → full-calibration protocol
        described in the class docstring.
        """
        all_pairs = [
            (solver, backend) for solver in self._solvers for backend in self._backends
        ]

        screened: list[tuple[LinearSolver, Backend, float]] = []
        unscreened: list[tuple[LinearSolver, Backend]] = []

        for solver, backend in all_pairs:
            screen = self._detect_screen_result(solver, backend, descriptor)
            if screen is None:
                unscreened.append((solver, backend))
            else:
                projected = screen.alpha * descriptor.n**solver.cost_exponent
                screened.append((solver, backend, projected))

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

    def _detect_screen_result(
        self,
        solver: LinearSolver,
        backend: Backend,
        descriptor: ProblemDescriptor,
    ) -> BenchmarkResult | None:
        """Return a BenchmarkResult at the smallest N where alpha has stabilized.

        Probes at N = 2, 4, 8, … doubling until consecutive alpha values satisfy
        alpha_prev / alpha_curr < self._alpha_stability.  That ratio converges to
        1.0 once compute dominates dispatch overhead; above the floor it is
        proportional to 2^p (one halving multiplies measured alpha by 2^p when
        overhead dominates).

        Returns None when no stable N is found below descriptor.n, signaling
        that this pair should skip the screen and go straight to full calibration.
        The returned result carries the alpha at the stable N and is reused
        directly by calibrate() to avoid a redundant measurement.
        """
        n = 2
        prev = self._benchmarker.measure(
            solver, backend, self._probe_descriptor(descriptor, n)
        )
        n = 4
        while n < descriptor.n:
            curr = self._benchmarker.measure(
                solver, backend, self._probe_descriptor(descriptor, n)
            )
            if prev.alpha / curr.alpha < self._alpha_stability:
                return prev
            prev = curr
            n *= 2
        return None

    @staticmethod
    def _probe_descriptor(descriptor: ProblemDescriptor, n: int) -> ProblemDescriptor:
        return ProblemDescriptor(
            n=n,
            g=descriptor.g,
            r=min(descriptor.r, n),
            tol=descriptor.tol,
            spectral_radius=descriptor.spectral_radius,
        )


__all__ = ["Autotuner", "SelectionResult"]
