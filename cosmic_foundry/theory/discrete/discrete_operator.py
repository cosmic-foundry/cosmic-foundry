"""DiscreteOperator ABC."""

from __future__ import annotations

from abc import abstractmethod
from typing import TypeVar

from cosmic_foundry.theory.continuous.differential_operator import DifferentialOperator
from cosmic_foundry.theory.discrete.discrete_field import DiscreteField
from cosmic_foundry.theory.foundation.numeric_function import NumericFunction

_V = TypeVar("_V")


class DiscreteOperator(NumericFunction[DiscreteField[_V], DiscreteField[_V]]):
    """The discrete analog of DifferentialOperator: Lₕ: DiscreteField → DiscreteField.

    A DiscreteOperator is the output of a Discretization — the Lₕ that makes
    the commutation diagram Lₕ ∘ Rₕ ≈ Rₕ ∘ L hold to the chosen approximation
    order.  It earns its class by two falsifiable claims:

        order               — integer convergence order; verified by Lane C
                              Taylor expansion in test_convergence_order.py
        continuous_operator — the DifferentialOperator L this instance
                              approximates; the mathematical specification
                              against which the convergence test measures error

    A DiscreteOperator is not constructed directly from stencil coefficients;
    it is produced by a Discretization, which derives continuous_operator
    automatically from the numerical flux.

    Required:
        order               — composite convergence order
        continuous_operator — the continuous operator this approximates
        __call__            — apply the operator: DiscreteField → DiscreteField
    """

    @property
    @abstractmethod
    def order(self) -> int:
        """Composite convergence order of the approximation scheme."""

    @property
    @abstractmethod
    def continuous_operator(self) -> DifferentialOperator:
        """The continuous operator this DiscreteOperator approximates."""

    @property
    @abstractmethod
    def stiffness_values(self) -> list[float]:
        """Non-zero stiffness matrix entries, one per (row, col) pair."""

    @property
    @abstractmethod
    def row_indices(self) -> list[int]:
        """Global row index for each entry in stiffness_values."""

    @property
    @abstractmethod
    def col_indices(self) -> list[int]:
        """Global column index for each entry in stiffness_values."""


__all__ = ["DiscreteOperator"]
