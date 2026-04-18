"""Record ABC and concrete record types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class Record(ABC):
    """Abstract base for all record types: lightweight immutable value objects
    that are *about* the simulation rather than *being* simulation state.

    Records are internal objects produced or consumed at the semantic layer —
    summaries, identifiers, and provenance metadata. They are distinct from
    Fields (which ARE simulation state) and from the external representations
    (bytes, files) that Sources and Sinks translate to and from.

    Every Record must be serializable to a plain dict.
    """

    @abstractmethod
    def as_dict(self) -> dict[str, Any]:
        """Return a plain-dict representation of this record."""


@dataclass(frozen=True)
class ComponentId(Record):
    """Opaque integer identifier for a named simulation component.

    Used wherever a typed, hashable, serializable integer key is needed —
    mesh blocks, field segments, and any future entity type.  A single class
    avoids redundant id types for concepts that are structurally identical.
    """

    value: int

    def as_dict(self) -> dict[str, Any]:
        return {"value": self.value}


class Placement(Record):
    """Maps each ``ComponentId`` to the process rank that owns it.

    ``Placement`` carries no physical meaning and no kernel-lowering logic.
    It is the sole authoritative source for process/device ownership within
    a composite ``PatchFunction``.
    """

    def __init__(self, owners: Mapping[ComponentId, int]) -> None:
        if not owners:
            msg = "Placement must register at least one segment"
            raise ValueError(msg)
        for sid, rank in owners.items():
            if rank < 0:
                msg = f"Process rank must be non-negative; got rank={rank} for {sid!r}"
                raise ValueError(msg)
        self._owners: dict[ComponentId, int] = dict(owners)

    def owner(self, segment_id: ComponentId) -> int:
        """Return the rank that owns *segment_id*."""
        try:
            return self._owners[segment_id]
        except KeyError:
            msg = f"ComponentId {segment_id!r} is not registered in this Placement"
            raise KeyError(msg) from None

    def segments_for_rank(self, rank: int) -> frozenset[ComponentId]:
        """Return the set of ComponentIds owned by *rank*."""
        return frozenset(sid for sid, r in self._owners.items() if r == rank)

    @property
    def component_ids(self) -> frozenset[ComponentId]:
        """Return all ComponentIds registered in this Placement."""
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
class Array(Record, Generic[T]):
    """A finite indexed family of elements with distributed ownership.

    Mathematically: a function I → T where I = {ComponentId(0), …, ComponentId(n-1)},
    together with a Placement recording which process rank owns each element.

    This is the general container for structured collections across the
    simulation: Array[Patch] represents a partitioned spatial domain;
    Array[PatchFunction] represents a distributed discrete field.
    """

    elements: tuple[T, ...]
    placement: Placement

    def __post_init__(self) -> None:
        expected = frozenset(ComponentId(i) for i in range(len(self.elements)))
        if self.placement.component_ids != expected:
            msg = (
                f"Placement component IDs {self.placement.component_ids} "
                f"do not match Array indices {expected}"
            )
            raise ValueError(msg)

    def __getitem__(self, component_id: ComponentId) -> T:
        return self.elements[component_id.value]

    def local(self, rank: int) -> tuple[T, ...]:
        """Return the elements owned by *rank*, in index order."""
        local_ids = self.placement.segments_for_rank(rank)
        return tuple(
            self.elements[cid.value]
            for cid in sorted(local_ids, key=lambda cid: cid.value)
        )

    def as_dict(self) -> dict[str, Any]:
        return {"n": len(self.elements), "placement": self.placement.as_dict()}


__all__ = [
    "Array",
    "ComponentId",
    "Placement",
    "Record",
]
