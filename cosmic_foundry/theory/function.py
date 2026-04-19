"""Function ABC."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Function(ABC):
    """Abstract base for all function classes: f: A × Θ → B.

    Every concrete Function subclass carries a ``Function:`` block in its
    class docstring specifying domain, codomain, operator, Θ, and
    approximation order p.  Subclasses that carry no parameters should use
    ``@dataclass(frozen=True)`` so that instances are hashable.
    """

    @abstractmethod
    def execute(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the function and return the result."""


__all__ = [
    "Function",
]
