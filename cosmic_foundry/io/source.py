"""Source ABC."""

from __future__ import annotations

from cosmic_foundry.theory.function import Function


class Source(Function):
    """Abstract base for all source classes: R: external state → B.

    Every concrete Source subclass carries a ``Source:`` block in its class
    docstring specifying the external state consumed (origin) and the value
    produced.  Subclasses that carry no parameters should use
    ``@dataclass(frozen=True)`` so that instances are hashable.
    """


__all__ = [
    "Source",
]
