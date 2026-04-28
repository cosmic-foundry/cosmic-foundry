"""DifferentialOperator: abstract operator mapping differential forms."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, TypeVar

import sympy

from cosmic_foundry.theory.continuous.differential_form import (
    DifferentialForm,
    ZeroForm,
)
from cosmic_foundry.theory.continuous.manifold import Manifold
from cosmic_foundry.theory.foundation.function import Function

_D = TypeVar("_D", bound=DifferentialForm)
_C = TypeVar("_C", bound=DifferentialForm)


class DifferentialOperator(Function[_D, _C]):
    """Abstract base for differential operators L: Ω^p → Ω^q.

    A differential operator of order r maps differential forms of one degree
    to differential forms of (possibly different) degree on the same manifold.
    The domain and codomain form degrees are determined by the TypeVar bounds
    _D and _C; concrete subclasses earn their class by fixing these types:

        d: OneForm → TwoForm      (exterior derivative, order 1)
        ∇·: OneForm → ZeroForm    (divergence, order 1)
        Δ: ZeroForm → ZeroForm    (Laplace–Beltrami, order 2)

    Required:
        manifold — the manifold on which this operator acts
        order    — the order of differentiation
    """

    @property
    @abstractmethod
    def manifold(self) -> Manifold:
        """The manifold on which this operator acts."""

    @property
    @abstractmethod
    def order(self) -> int:
        """The order of differentiation."""


class DivergenceComposition(DifferentialOperator[Any, ZeroForm[Any]]):
    """∇·F: the divergence of a DifferentialOperator F mapping ZeroForm → OneForm.

    DivergenceComposition(F)(φ) computes ∇·(F(φ)) by differentiating the
    components of the OneForm F(φ) with respect to the coordinate symbols and
    returning the sum as a ZeroForm.  The order of the composition is F.order + 1.

    This is the continuous Laplacian building block: wrapping DiffusionOperator
    (which returns -∇φ as a OneForm) gives the negative Laplacian ∇·(-∇φ) = -∇²φ.
    """

    def __init__(self, flux_op: DifferentialOperator) -> None:
        self._flux_op = flux_op

    @property
    def manifold(self) -> Any:
        return self._flux_op.manifold

    @property
    def order(self) -> int:
        return self._flux_op.order + 1

    def __call__(self, phi: Any) -> ZeroForm[Any]:
        one_form = self._flux_op(phi)
        div: sympy.Expr = sum(
            (
                sympy.diff(one_form.component(i), one_form.symbols[i])
                for i in range(len(one_form.symbols))
            ),
            sympy.Integer(0),
        )
        return ZeroForm(phi.manifold, div, phi.symbols)


__all__ = ["DifferentialOperator", "DivergenceComposition"]
