"""Source ABC."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Source(ABC):
    """Abstract base for all source classes: R: external state → B.

    Every concrete Source subclass carries a ``Source:`` block in its class
    docstring specifying the external state consumed (origin) and the value
    produced.  Subclasses that carry no parameters should use
    ``@dataclass(frozen=True)`` so that instances are hashable.
    """

    @abstractmethod
    def execute(self, *args: Any, **kwargs: Any) -> Any:
        """Read from external state and return the result."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate to execute()."""
        return self.execute(*args, **kwargs)


__all__ = [
    "Source",
]
