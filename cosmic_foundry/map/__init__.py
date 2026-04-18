"""Map ABC."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Map(ABC):
    """Abstract base for all map classes: M: A × Θ → B.

    Every concrete Map subclass carries a ``Map:`` block in its class
    docstring specifying domain, codomain, operator, Θ, and approximation
    order p.  Subclasses that carry no parameters should use
    ``@dataclass(frozen=True)`` so that instances are hashable.
    """

    @abstractmethod
    def execute(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the map and return the result."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate to execute(); lets a Map instance be used as a callable."""
        return self.execute(*args, **kwargs)


__all__ = [
    "Map",
]
