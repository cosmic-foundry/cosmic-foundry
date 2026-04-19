"""Record ABC: serializable manifest value objects."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Record(ABC):
    """Abstract base for manifest value objects: lightweight immutable objects
    that describe provenance, identity, and validation metadata.

    Every Record must be serializable to a plain dict.
    """

    @abstractmethod
    def as_dict(self) -> dict[str, Any]:
        """Return a plain-dict representation of this record."""


__all__ = ["Record"]
