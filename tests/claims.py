"""Shared base classes, calibration types, and budget constants for test claims."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

C = TypeVar("C")

# Single source of truth for the convergence-rate test-suite walltime budget
# and the over-budget tolerance.  Both numbers feed three independent enforcement
# layers — _convergence_n_max sizing, the per-claim end-of-check() assertion,
# and the pytest session_timeout backstop — so changing them here updates every
# layer consistently.
# Budget can be set via COSMIC_FOUNDRY_TEST_BUDGET_S (float, seconds),
# COSMIC_FOUNDRY_TEST_FAST=1 (shorthand for 5s), or defaults to 60s.
_BUDGET_ENV = os.environ.get("COSMIC_FOUNDRY_TEST_BUDGET_S")
if os.environ.get("COSMIC_FOUNDRY_TEST_FAST") == "1":
    MAX_WALLTIME_S = 5.0
elif _BUDGET_ENV is not None:
    MAX_WALLTIME_S = float(_BUDGET_ENV)
else:
    MAX_WALLTIME_S = 60.0
BUDGET_TOLERANCE = 1.1


class Claim(ABC):
    """Base for static correctness claims that do not depend on calibration."""

    @property
    @abstractmethod
    def description(self) -> str: ...

    @abstractmethod
    def check(self) -> None: ...


class CalibratedClaim(ABC, Generic[C]):
    """Base for claims whose verification requires runtime calibration data.

    The type parameter C is the calibration type.  Bind it concretely in each
    claim family, e.g. CalibratedClaim[float] for FMA-rate-calibrated claims.
    """

    @property
    @abstractmethod
    def description(self) -> str: ...

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
class ConvergenceCalibration:
    """Per-device, per-solver calibration data for convergence-rate claims.

    Holds backend instances and a cost coefficient α per (device, solver class)
    pair.  Each α is calibrated by a real timed solve on the target device, so
    _convergence_n_max(α, solver.cost_exponent, budget) reflects actual solver
    throughput on that hardware — not a modeled roofline, and not shared
    across solvers with different complexity (Jacobi is O(N^4), LU is O(N^3),
    so a single α would mis-size at least one of them).

    cpu_alphas is keyed by LinearSolver subclass (e.g. DenseJacobiSolver).
    gpu_* fields are None when no GPU device is available.
    """

    cpu_backend: Any
    gpu_backend: Any | None
    cpu_alphas: dict[type, float]
    gpu_alphas: dict[type, float] | None
