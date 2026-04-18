"""Record ABC and concrete record types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


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


__all__ = [
    "ComponentId",
    "Record",
]
