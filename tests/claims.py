"""Shared base classes and calibration types for all test claims."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

C = TypeVar("C")


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
class JaxCalibration:
    """Post-JIT JAX FMA throughput for CPU and (optionally) GPU.

    Both rates are measured with jax.jit and block_until_ready so they
    reflect steady-state XLA throughput rather than eager-dispatch latency.
    gpu_fma_rate is None when no GPU device is available.
    """

    cpu_fma_rate: float
    gpu_fma_rate: float | None
