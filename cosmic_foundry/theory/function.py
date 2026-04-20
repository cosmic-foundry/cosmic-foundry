"""Function ABC."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

D = TypeVar("D")  # Domain
C = TypeVar("C")  # Codomain


class Function(ABC, Generic[D, C]):
    """A callable mapping domain D to codomain C.

    Subclasses parameterize D (domain type) and C (codomain type), making
    the mathematical contract explicit in the type signature.

    Example:
        class WriteArray(Sink[tuple[Path, Any]]): ...
        class LoadSchema(Source[str, dict[str, Any]]): ...
        class Field(Function[Point, Vector]): ...

    All concrete Function instances should use ``@dataclass(frozen=True)``
    for hashability when they carry no mutable state.
    """

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> C: ...


__all__ = [
    "Function",
]
