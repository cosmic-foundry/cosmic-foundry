"""Homeomorphism ABC."""

from __future__ import annotations

from abc import abstractmethod
from typing import Generic, TypeVar

from cosmic_foundry.foundation.function import Function
from cosmic_foundry.foundation.topological_space import TopologicalSpace

D = TypeVar("D")
C = TypeVar("C")


class Homeomorphism(Function[D, C], Generic[D, C]):
    """A bicontinuous bijection between topological spaces: φ: U → V.

    A homeomorphism is continuous, bijective, and has a continuous inverse.
    Two topological spaces connected by a homeomorphism are topologically
    indistinguishable.

    Required:
        domain   — the source topological space U
        codomain — the target topological space V
        inverse  — the continuous inverse φ⁻¹: V → U
    """

    @property
    @abstractmethod
    def domain(self) -> TopologicalSpace:
        """The source topological space U."""

    @property
    @abstractmethod
    def codomain(self) -> TopologicalSpace:
        """The target topological space V."""

    @property
    @abstractmethod
    def inverse(self) -> Homeomorphism[C, D]:
        """The continuous inverse φ⁻¹: V → U."""


__all__ = ["Homeomorphism"]
