"""Sink ABC."""

from __future__ import annotations

from typing import TypeVar

from cosmic_foundry.theory.function import Function

D = TypeVar("D")  # Domain


class Sink(Function[D, None]):
    """Abstract base for all sink classes: D → external state (None).

    Codomain is always None; the effect is external side effects.
    Subclasses bind D to a specific domain type.
    """


__all__ = [
    "Sink",
]
