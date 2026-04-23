"""SymbolicFunction ABC."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Generic, TypeVar

import sympy

from cosmic_foundry.theory.foundation.function import Function

D = TypeVar("D")
C = TypeVar("C")


class SymbolicFunction(Function[D, C], Generic[D, C]):
    """A Function defined by a SymPy expression.

    __call__ is derived: it substitutes positional args into expr
    according to the canonical ordering declared by symbols.

    Required:
        expr    — SymPy expression for this function's output
        symbols — ordered free symbols; defines argument order for __call__
    """

    @property
    @abstractmethod
    def expr(self) -> sympy.Expr:
        """SymPy expression defining this function's output."""

    @property
    @abstractmethod
    def symbols(self) -> tuple[sympy.Symbol, ...]:
        """Ordered free symbols; canonical argument order for __call__."""

    def __call__(self, *args: Any, **kwargs: Any) -> sympy.Expr:
        return self.expr.subs(zip(self.symbols, args, strict=False))


__all__ = ["SymbolicFunction"]
