"""DifferentialOperator: abstract operator mapping differential forms."""

from __future__ import annotations

from abc import abstractmethod
from typing import TypeVar

from cosmic_foundry.theory.continuous.differential_form import DifferentialForm
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


__all__ = ["DifferentialOperator"]
