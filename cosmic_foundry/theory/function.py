"""Function ABC."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Function(ABC):
    """A callable with a formal mathematical contract.

    Subclasses declare: domain, codomain, parameters Θ, approximation order p.
    All concrete Function instances should use ``@dataclass(frozen=True)``
    for hashability when they carry no mutable state.
    """

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


__all__ = [
    "Function",
]
