"""Source ABC."""

from __future__ import annotations

from cosmic_foundry.theory.function import Function


class Source(Function):
    """Abstract base for all source classes: external state → B.

    Concrete subclasses carry a ``Source:`` block specifying the external
    state consumed (origin) and the value produced (codomain).
    """


__all__ = [
    "Source",
]
