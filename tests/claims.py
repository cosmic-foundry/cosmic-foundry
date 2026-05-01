"""Shared base classes, calibration types, and budget constants for test claims."""

from __future__ import annotations

import math
import os
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import pytest

C = TypeVar("C")

# ── Solver convergence budget ─────────────────────────────────────────────────
# Controls mesh-refinement N_max sizing for discrete-operator convergence claims.
# Set via CF_SOLVER_CONVERGENCE_BUDGET_S (default 5 s locally, 60 s in CI).
SOLVER_CONVERGENCE_BUDGET_S: float = float(
    os.environ.get("CF_SOLVER_CONVERGENCE_BUDGET_S", "5.0")
)

# ── Time-integrator test budgets ──────────────────────────────────────────────
# INTEGRATOR_CLAIM_BUDGET_S  : per-claim wall-time cap for the halving loop in
#                test_time_integrators.py.  Governs how many dt halvings are
#                attempted before moving on.  Set via
#                CF_INTEGRATOR_CLAIM_BUDGET_S (default 1 s locally and in CI).
# INTEGRATOR_SESSION_BUDGET_S : expected total wall time for the entire
#                test_time_integrators.py session (convergence claims +
#                NSE claims + behavior checks).  Feeds the session-level
#                timeout alongside SOLVER_CONVERGENCE_BUDGET_S so neither
#                suite's budget inflates the other's.  Set via
#                CF_INTEGRATOR_SESSION_BUDGET_S (default 30 s locally, 90 s in CI).
INTEGRATOR_CLAIM_BUDGET_S: float = float(
    os.environ.get("CF_INTEGRATOR_CLAIM_BUDGET_S", "1.0")
)
INTEGRATOR_SESSION_BUDGET_S: float = float(
    os.environ.get("CF_INTEGRATOR_SESSION_BUDGET_S", "30.0")
)

# ── Shared fixed overhead ─────────────────────────────────────────────────────
# Per-session overhead not covered by either convergence budget: performance-
# test calibration, structure/tensor tests, solver calibration (one probe per
# solver type per session), and GPU benchmark variability.
FIXED_SESSION_OVERHEAD_S: float = 40.0

# Tolerance multiplier on the total expected session time.
BUDGET_TOLERANCE: float = 1.1

# ── Per-claim walltime budget ────────────────────────────────────────────────
# Unified default gate for claims that are valuable but too expensive for normal
# local/CI runs.  Raise CF_CLAIM_WALLTIME_BUDGET_S for targeted stress runs.
CLAIM_WALLTIME_BUDGET_S: float = float(
    os.environ.get("CF_CLAIM_WALLTIME_BUDGET_S", "1.0")
)

# ── Batched-claim replay ─────────────────────────────────────────────────────
# Set to a failed lane's batch index to rerun only that lane as a CPU scalar
# debug case while preserving the original lane's parameterization.
BATCH_REPLAY_INDEX_ENV = "CF_TEST_BATCH_INDEX"

# GPU calibration is trusted only when its compute-bound Tensor roofline is
# materially faster than the CPU JIT roofline.  Otherwise GPU tests skip and
# ExecutionPlan stays on CPU so module tests do not run on a misconfigured or
# CPU-fallback device.
DEVICE_GPU_CPU_MIN_SPEEDUP = 2.0


def invalid_fma_rate_reason(label: str, rate: float | None) -> str | None:
    if rate is None:
        return f"{label} calibration did not produce a rate"
    if not math.isfinite(rate) or rate <= 0.0:
        return f"{label} calibration produced invalid rate {rate!r}"
    return None


def gpu_trust_skip_reason(cpu_rate: float, gpu_rate: float | None) -> str | None:
    """Return None when GPU calibration is trusted, otherwise the skip reason."""
    invalid_cpu = invalid_fma_rate_reason("CPU", cpu_rate)
    if invalid_cpu is not None:
        raise RuntimeError(invalid_cpu)
    invalid_gpu = invalid_fma_rate_reason("GPU", gpu_rate)
    if invalid_gpu is not None:
        return invalid_gpu
    assert gpu_rate is not None
    speedup = gpu_rate / cpu_rate
    if speedup < DEVICE_GPU_CPU_MIN_SPEEDUP:
        return (
            f"GPU roofline {gpu_rate:.2e} FMAs/s is only {speedup:.1f}x CPU "
            f"roofline {cpu_rate:.2e} FMAs/s; requires "
            f"{DEVICE_GPU_CPU_MIN_SPEEDUP:.1f}x"
        )
    return None


class Claim(ABC, Generic[C]):
    """Base for every test claim.

    Claims receive a module-defined calibration object.  Static claims use
    ``None`` as their trivial calibration so the dispatch contract remains
    uniform across correctness, convergence, and performance axes.
    """

    @property
    @abstractmethod
    def description(self) -> str: ...

    @property
    def expected_walltime_s(self) -> float:
        """Expected walltime for default local/CI eligibility."""
        return 1.0

    def skip_if_over_walltime_budget(self) -> None:
        """Skip claims whose declared walltime exceeds the active budget."""
        if self.expected_walltime_s > CLAIM_WALLTIME_BUDGET_S:
            pytest.skip(
                f"{self.description}: expected {self.expected_walltime_s:.1f}s "
                f"> walltime budget {CLAIM_WALLTIME_BUDGET_S:.1f}s"
            )

    @abstractmethod
    def check(self, calibration: C) -> None: ...


@dataclass(frozen=True)
class DeviceCalibration:
    """FMA throughput rooflines and backend instances for available compute devices.

    cpu_backend and cpu_fma_rate always refer to the CPU device.
    gpu_backend and gpu_fma_rate are None when no functional GPU backend is
    available (no device found, XLA/driver error during measurement, invalid
    timing result, or GPU roofline too close to CPU roofline).  gpu_skip_reason
    records the reason GPU execution is unavailable for skip messages.

    The backends stored here are the exact instances used during calibration;
    performance claims should use them for benchmarking so that the measured
    roofline and the claim workload run through the same code paths.
    """

    cpu_backend: Any
    gpu_backend: Any | None
    cpu_fma_rate: float
    gpu_fma_rate: float | None
    gpu_skip_reason: str | None = None


@dataclass(frozen=True)
class BatchedFailure:
    """Failure metadata for one lane of a batched claim."""

    claim: str
    device_kind: str
    batch_size: int
    batch_index: int
    method: str
    order: int | None
    problem: str
    parameters: Mapping[str, object]
    actual: object
    expected: object
    error: float
    tolerance: float

    def format(self) -> str:
        """Return a replay-ready assertion message for a failed batch lane."""
        order = "n/a" if self.order is None else str(self.order)
        params = ", ".join(
            f"{name}={value!r}" for name, value in sorted(self.parameters.items())
        )
        return (
            f"{self.claim}/{self.device_kind}: batch={self.batch_size}, "
            f"batch_index={self.batch_index}, method={self.method}, "
            f"order={order}, problem={self.problem}, parameters={{ {params} }}, "
            f"actual={self.actual!r}, expected={self.expected!r}, "
            f"error={self.error:.3e} >= {self.tolerance:.3e}; "
            f"replay with {BATCH_REPLAY_INDEX_ENV}={self.batch_index}"
        )


@dataclass(frozen=True)
class ExecutionPlan:
    """Device choice and walltime budget for scalable Tensor-backed claims."""

    backend: Any
    device_kind: str
    claim_walltime_budget_s: float
    device_calibration: DeviceCalibration

    @property
    def fma_rate(self) -> float:
        if self.device_kind == "gpu":
            assert self.device_calibration.gpu_fma_rate is not None
            return self.device_calibration.gpu_fma_rate
        return self.device_calibration.cpu_fma_rate

    def extent_for(
        self,
        *,
        work_fmas: Callable[[int], float],
        min_extent: int,
        max_extent: int,
        label: str,
        safety: float = 0.5,
    ) -> int:
        """Largest monotone extent expected to fit the active claim budget."""
        if min_extent < 1:
            raise ValueError(f"{label}: min_extent must be >= 1")
        if max_extent < min_extent:
            raise ValueError(f"{label}: max_extent must be >= min_extent")
        if not 0.0 < safety <= 1.0:
            raise ValueError(f"{label}: safety must be in (0, 1]")

        budget_fmas = self.claim_walltime_budget_s * safety * self.fma_rate
        min_work = work_fmas(min_extent)
        if min_work <= 0.0:
            raise ValueError(f"{label}: work estimate must be positive")
        if min_work > budget_fmas:
            pytest.skip(
                f"{label}: smallest extent {min_extent} needs "
                f"{min_work:.3g} FMAs > budget {budget_fmas:.3g} FMAs "
                f"({self.claim_walltime_budget_s:.3g}s on {self.device_kind})"
            )

        lo = min_extent
        hi = max_extent
        while lo < hi:
            mid = (lo + hi + 1) // 2
            mid_work = work_fmas(mid)
            if mid_work <= 0.0:
                raise ValueError(f"{label}: work estimate must be positive")
            if mid_work <= budget_fmas:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def batch_size_for(
        self,
        *,
        fmas_per_case: float,
        min_batch: int,
        max_batch: int,
        safety: float = 0.5,
    ) -> int:
        """Largest batch expected to fit the claim budget, clamped to bounds."""
        if fmas_per_case <= 0.0:
            raise ValueError("batch: fmas_per_case must be positive")
        return self.extent_for(
            work_fmas=lambda batch: batch * fmas_per_case,
            min_extent=min_batch,
            max_extent=max_batch,
            label="batch",
            safety=safety,
        )

    def problem_size_for(
        self,
        *,
        work_fmas: Callable[[int], float],
        min_size: int,
        max_size: int,
        label: str = "problem_size",
        safety: float = 0.5,
    ) -> int:
        """Largest problem size expected to fit the claim budget."""
        return self.extent_for(
            work_fmas=work_fmas,
            min_extent=min_size,
            max_extent=max_size,
            label=label,
            safety=safety,
        )

    def refinement_count_for(
        self,
        *,
        work_fmas: Callable[[int], float],
        min_refinements: int,
        max_refinements: int,
        label: str = "refinement_count",
        safety: float = 0.5,
    ) -> int:
        """Largest refinement count expected to fit the claim budget."""
        return self.extent_for(
            work_fmas=work_fmas,
            min_extent=min_refinements,
            max_extent=max_refinements,
            label=label,
            safety=safety,
        )

    def replay_batch_index(self, batch_size: int, *, label: str) -> int | None:
        """Requested scalar replay lane, or None when full batching is active."""
        raw = os.environ.get(BATCH_REPLAY_INDEX_ENV)
        if raw is None:
            return None
        try:
            index = int(raw)
        except ValueError as exc:
            raise ValueError(
                f"{label}: {BATCH_REPLAY_INDEX_ENV} must be an integer"
            ) from exc
        if index < 0 or index >= batch_size:
            raise ValueError(
                f"{label}: {BATCH_REPLAY_INDEX_ENV}={index} outside "
                f"batch index range [0, {batch_size})"
            )
        return index

    def batch_indices_for(self, batch_size: int, *, label: str) -> tuple[int, ...]:
        """Full batch indices, or the single replay index from the environment."""
        replay_index = self.replay_batch_index(batch_size, label=label)
        if replay_index is not None:
            return (replay_index,)
        return tuple(range(batch_size))
