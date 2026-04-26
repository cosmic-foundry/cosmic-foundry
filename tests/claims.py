"""Shared base classes for all test claims."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Claim(ABC):
    """Base for static correctness claims that do not depend on machine speed."""

    @property
    @abstractmethod
    def description(self) -> str: ...

    @abstractmethod
    def check(self) -> None: ...


class CalibratedClaim(ABC):
    """Base for claims whose verification depends on the machine's FMA rate."""

    @property
    @abstractmethod
    def description(self) -> str: ...

    @abstractmethod
    def check(self, fma_rate: float) -> None: ...
