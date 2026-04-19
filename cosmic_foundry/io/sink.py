"""Sink ABC."""

from __future__ import annotations

from cosmic_foundry.theory.function import Function


class Sink(Function):
    """Abstract base for all sink classes: S: A → external state.

    Every concrete Sink subclass carries a ``Sink:`` block in its class
    docstring specifying the domain consumed and the external effect produced.
    Subclasses that carry no parameters should use
    ``@dataclass(frozen=True)`` so that instances are hashable.
    """


__all__ = [
    "Sink",
]
