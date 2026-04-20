"""Sink ABC."""

from __future__ import annotations

from cosmic_foundry.theory.function import Function


class Sink(Function):
    """Abstract base for all sink classes: A → external state.

    Concrete subclasses carry a ``Sink:`` block specifying the domain,
    codomain (usually None), and external effect produced.
    """


__all__ = [
    "Sink",
]
