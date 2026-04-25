"""ConvergenceOracle protocol."""

from __future__ import annotations

from typing import Any, Protocol, TypeVar, runtime_checkable

import sympy

T = TypeVar("T", covariant=True)


@runtime_checkable
class ConvergenceOracle(Protocol[T]):
    """Test-side oracle for a convergent class T.

    Provides a manufactured-solution error expression for a given instance
    and symbolic mesh spacing.  The oracle knows the physics; the test does
    not.  Any class claiming `order: int` convergence needs exactly one
    oracle registered in tests/support/oracles/.
    """

    def instances(self) -> list[Any]:
        """All parameter combinations to test."""
        ...

    def error(self, instance: Any, h: sympy.Symbol) -> sympy.Expr:
        """Manufactured-solution error as a sympy expression in h.

        Returns numerical − exact for a smooth test field on a mesh with
        symbolic spacing h, using the interior face / cell / degree of
        freedom appropriate for the instance.
        """
        ...


__all__ = ["ConvergenceOracle"]
