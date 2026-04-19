"""Sink ABC."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Sink(ABC):
    """Abstract base for all sink classes: S: A → external state.

    Every concrete Sink subclass carries a ``Sink:`` block in its class
    docstring specifying the domain consumed and the external effect produced.
    Subclasses that carry no parameters should use
    ``@dataclass(frozen=True)`` so that instances are hashable.
    """

    @abstractmethod
    def execute(self, *args: Any, **kwargs: Any) -> Any:
        """Consume input and materialise it into external state."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate to execute()."""
        return self.execute(*args, **kwargs)


__all__ = [
    "Sink",
]
