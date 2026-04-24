"""NumericFunction ABC."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Generic, TypeVar

from cosmic_foundry.theory.foundation.function import Function

D = TypeVar("D")
C = TypeVar("C")


class NumericFunction(Function[D, C], Generic[D, C]):
    """A Function implemented procedurally, with an optional symbolic counterpart.

    Declares the refinement relationship to a SymbolicFunction where the
    numeric implementation is known to implement a named analytic form.

    Required:
        __call__ — numerical evaluation

    Optional:
        symbolic — the SymbolicFunction this implements, if declared;
                   None means no symbolic counterpart is known or applicable
    """

    @abstractmethod
    def __call__(self, x: D) -> C: ...

    @property
    def symbolic(self) -> Any:
        """The SymbolicFunction this numerically implements, or None."""
        return None


__all__ = ["NumericFunction"]
