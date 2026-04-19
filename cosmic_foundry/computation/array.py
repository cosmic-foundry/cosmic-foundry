"""Distributed indexed family: Array[T] and Placement."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class Placement:
    """Maps each integer component index to the process rank that owns it.

    ``Placement`` carries no physical meaning and no kernel-lowering logic.
    It is the sole authoritative source for process/device ownership within
    a composite ``Array[T]``.
    """

    def __init__(self, owners: Mapping[int, int]) -> None:
        if not owners:
            msg = "Placement must register at least one segment"
            raise ValueError(msg)
        for sid, rank in owners.items():
            if rank < 0:
                msg = f"Process rank must be non-negative; got rank={rank} for {sid}"
                raise ValueError(msg)
        self._owners: dict[int, int] = dict(owners)

    def owner(self, segment_id: int) -> int:
        """Return the rank that owns *segment_id*."""
        try:
            return self._owners[segment_id]
        except KeyError:
            msg = f"Component index {segment_id!r} is not registered in this Placement"
            raise KeyError(msg) from None

    def segments_for_rank(self, rank: int) -> frozenset[int]:
        """Return the set of component indices owned by *rank*."""
        return frozenset(sid for sid, r in self._owners.items() if r == rank)

    @property
    def component_ids(self) -> frozenset[int]:
        """Return all component indices registered in this Placement."""
        return frozenset(self._owners.keys())

    def as_dict(self) -> dict[str, Any]:
        return {str(k): v for k, v in self._owners.items()}

    def __repr__(self) -> str:
        return f"Placement({dict(self._owners)!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Placement):
            return NotImplemented
        return self._owners == other._owners

    def __hash__(self) -> int:
        return hash(tuple(sorted(self._owners.items())))


@dataclass(frozen=True)
class Array(Generic[T]):
    """A finite indexed family of elements with distributed ownership.

    Mathematically: a function I → T where I = {0, 1, …, n-1},
    together with a Placement recording which process rank owns each element.

    This is the general container for structured collections across the
    simulation: Array[Patch] represents a partitioned spatial domain;
    Array[jax.Array] represents a distributed discrete field (MultiFab pattern).
    """

    elements: tuple[T, ...]
    placement: Placement

    def __post_init__(self) -> None:
        expected = frozenset(range(len(self.elements)))
        if self.placement.component_ids != expected:
            msg = (
                f"Placement component indices {self.placement.component_ids} "
                f"do not match Array indices {expected}"
            )
            raise ValueError(msg)

    def __getitem__(self, index: int) -> T:
        return self.elements[index]

    def local(self, rank: int) -> tuple[T, ...]:
        """Return the elements owned by *rank*, in index order."""
        local_ids = self.placement.segments_for_rank(rank)
        return tuple(self.elements[i] for i in sorted(local_ids))

    def as_dict(self) -> dict[str, Any]:
        return {"n": len(self.elements), "placement": self.placement.as_dict()}


__all__ = [
    "Array",
    "Placement",
]
