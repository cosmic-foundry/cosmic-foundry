"""Record ABC: serializable value objects about the simulation."""

from __future__ import annotations

from abc import ABC, abstractmethod
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


__all__ = ["Record"]
