"""SymbolicFunction: a Function defined by a SymPy expression, evaluated at Point[M]."""

from __future__ import annotations

from abc import abstractmethod
from typing import Generic, TypeVar

import sympy

from cosmic_foundry.theory.continuous.manifold import Point
from cosmic_foundry.theory.foundation.function import Function

M = TypeVar("M")  # Manifold type
C = TypeVar("C")  # Codomain value type


class SymbolicFunction(Function[Point[M], C], Generic[M, C]):
    """A Function defined by a SymPy expression, evaluated at a typed Point[M].

    __call__ is derived: it verifies that the point's chart matches the
    field's coordinate symbols, then substitutes point.coords into expr.
    The chart check is skipped for constant fields (empty symbols) since
    they are coordinate-independent.

    Required:
        expr    — SymPy expression for this function's output
        symbols — ordered coordinate symbols; must match point.chart.symbols
    """

    @property
    @abstractmethod
    def expr(self) -> sympy.Expr:
        """SymPy expression defining this function's output."""

    @property
    @abstractmethod
    def symbols(self) -> tuple[sympy.Symbol, ...]:
        """Ordered coordinate symbols; must match point.chart.symbols on evaluation."""

    def __call__(self, point: Point[M]) -> C:
        if self.symbols and point.chart.symbols != self.symbols:
            raise ValueError(
                f"Chart mismatch: point uses chart with symbols "
                f"{point.chart.symbols}, but field expects {self.symbols}"
            )
        if len(point.coords) != len(self.symbols):
            raise ValueError(
                f"Coordinate count mismatch: point has {len(point.coords)} "
                f"coordinate(s) but field expects {len(self.symbols)}"
            )
        return self.expr.subs(zip(self.symbols, point.coords, strict=False))  # type: ignore[no-any-return]


__all__ = ["SymbolicFunction"]
