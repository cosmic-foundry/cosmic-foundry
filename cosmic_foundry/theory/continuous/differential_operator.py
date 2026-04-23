"""DifferentialOperator: abstract operator mapping fields to fields."""

from __future__ import annotations

from abc import abstractmethod

from cosmic_foundry.theory.continuous.field import Field
from cosmic_foundry.theory.continuous.manifold import Manifold
from cosmic_foundry.theory.foundation.function import Function


class DifferentialOperator(Function[Field, Field]):
    """Abstract base for differential operators L: Field → Field.

    A differential operator of order r maps smooth fields on a manifold
    M to smooth fields on the same manifold, using up to r-th order
    partial derivatives.  Concrete examples:

        order 1 — gradient, divergence, curl, exterior derivative d
        order 2 — Laplacian, Laplace–Beltrami operator

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
