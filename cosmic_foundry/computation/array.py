"""Finite indexed family: Array[T]."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from cosmic_foundry.theory.indexed_family import IndexedFamily

T = TypeVar("T")


@dataclass(frozen=True)
class Array(IndexedFamily, Generic[T]):
    """A finite indexed family — a function {0, 1, …, n-1} → T.

    IndexedFamily:
        domain   — i ∈ {0, 1, …, n-1} ⊂ ℤ
        codomain — T
        operator — i ↦ elements[i]

    This is the general container for structured collections across the
    simulation: Array[Patch] represents a partitioned spatial domain;
    Array[jax.Array] represents a distributed discrete field (MultiFab pattern).
    """

    elements: tuple[T, ...]

    def __post_init__(self) -> None:
        if not self.elements:
            msg = "Array must have at least one element"
            raise ValueError(msg)

    def __getitem__(self, index: int) -> T:
        return self.elements[index]

    def __len__(self) -> int:
        return len(self.elements)

    def as_dict(self) -> dict[str, Any]:
        return {"n": len(self.elements)}


__all__ = [
    "Array",
]
