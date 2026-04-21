"""InvertibleFunction ABC."""

from __future__ import annotations

from abc import abstractmethod
from typing import Generic, TypeVar

from cosmic_foundry.foundation.function import Function

D = TypeVar("D")
C = TypeVar("C")


class InvertibleFunction(Function[D, C], Generic[D, C]):
    """A bijection f: A → B admitting an inverse g: B → A.

    An invertible function (bijection) has a two-sided inverse: g∘f = id_A
    and f∘g = id_B.  This is a purely set-theoretic concept — no topology,
    metric, or smooth structure is assumed.

    Subclasses add structure:
    - Bijection between topological spaces with continuous inverse — Homeomorphism

    Required:
        domain   — the source A
        codomain — the target B
        inverse  — the two-sided inverse g: B → A
    """

    @property
    @abstractmethod
    def domain(self) -> object:
        """The source of this function."""

    @property
    @abstractmethod
    def codomain(self) -> object:
        """The target of this function."""

    @property
    @abstractmethod
    def inverse(self) -> InvertibleFunction[C, D]:
        """The two-sided inverse g: B → A."""


__all__ = ["InvertibleFunction"]
