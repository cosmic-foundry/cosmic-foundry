"""DiscreteOperator ABC."""

from __future__ import annotations

from abc import abstractmethod
from typing import TypeVar

from cosmic_foundry.theory.continuous.differential_operator import DifferentialOperator
from cosmic_foundry.theory.foundation.numeric_function import NumericFunction

_In = TypeVar("_In")
_Out = TypeVar("_Out")


class DiscreteOperator(NumericFunction[_In, _Out]):
    """Discrete operator with convergence order and continuous counterpart.

    Parameterized by input and output types so subclasses can encode the
    cochain shape: cell → cell (Discretization), cell → face (NumericalFlux),
    etc.

    The Lₕ that makes the commutation diagram Lₕ ∘ Rₕ ≈ Rₕ ∘ L hold to the
    chosen approximation order.  It earns its class by two falsifiable claims:

        order               — integer convergence order; verified by Lane C
                              Taylor expansion in test_convergence_order.py
        continuous_operator — the DifferentialOperator L this instance
                              approximates; the mathematical specification
                              against which the convergence test measures error

    Concrete subclasses are Discretization subclasses (e.g.
    DivergenceFormDiscretization), which derive continuous_operator
    automatically from the numerical flux.

    Required:
        order               — composite convergence order
        continuous_operator — the continuous operator this approximates
        __call__            — apply the operator
    """

    @property
    @abstractmethod
    def order(self) -> int:
        """Composite convergence order of the approximation scheme."""

    @property
    @abstractmethod
    def continuous_operator(self) -> DifferentialOperator:
        """The continuous operator this DiscreteOperator approximates."""


__all__ = ["DiscreteOperator"]
