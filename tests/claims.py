"""Shared base classes, calibration types, and budget constants for test claims."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
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
    available (no device found, or XLA/driver error during measurement).

    The backends stored here are the exact instances used during calibration;
    performance claims should use them for benchmarking so that the measured
    roofline and the claim workload run through the same code paths.
    """

    cpu_backend: Any
    gpu_backend: Any | None
    cpu_fma_rate: float
    gpu_fma_rate: float | None


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

    def batch_size_for(
        self,
        *,
        fmas_per_case: float,
        min_batch: int,
        max_batch: int,
        safety: float = 0.5,
    ) -> int:
        """Largest batch expected to fit the claim budget, clamped to bounds."""
        budget = self.claim_walltime_budget_s * safety
        if fmas_per_case <= 0.0:
            return min_batch
        estimated = int(budget * self.fma_rate / fmas_per_case)
        return max(min_batch, min(max_batch, estimated))
