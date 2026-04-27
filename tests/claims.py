"""Shared base classes, calibration types, and budget constants for test claims."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

C = TypeVar("C")

# Convergence-test walltime budget.  Controls N_max sizing only; set via
# COSMIC_FOUNDRY_TEST_BUDGET_S env var (default 5s locally, 30s in CI).
MAX_WALLTIME_S = float(os.environ.get("COSMIC_FOUNDRY_TEST_BUDGET_S", "5.0"))

# Fixed per-session overhead not covered by MAX_WALLTIME_S: performance-test
# calibration (~7s), structure/tensor tests (~4s), convergence calibration (~2s).
FIXED_SESSION_OVERHEAD_S = 20.0

# Tolerance multiplier on the total expected session time.
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
