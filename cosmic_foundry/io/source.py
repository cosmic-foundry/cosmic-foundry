"""Source ABC."""

from __future__ import annotations

from typing import TypeVar

from cosmic_foundry.theory.function import Function

D = TypeVar("D")  # Domain
C = TypeVar("C")  # Codomain


class Source(Function[D, C]):
    """Abstract base for all source classes: external state (D) → C.

    Subclasses bind D (external state/query) and C (output type).
    """


__all__ = [
    "Source",
]
