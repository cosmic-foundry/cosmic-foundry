"""NumericalFlux ABC."""

from __future__ import annotations

from typing import TypeVar

from cosmic_foundry.theory.discrete.discrete_operator import DiscreteOperator

_V = TypeVar("_V")


class NumericalFlux(DiscreteOperator[_V]):
    """Discrete approximation of a continuous flux operator.

    NumericalFlux narrows DiscreteOperator to flux approximations: concrete
    subclasses in the physics layer implement specific physical fluxes
    (diffusive, advective) that discrete schemes (FVM, FD) consume via their
    respective assembly patterns.

    Required (inherited from DiscreteOperator):
        order               — convergence order of the approximation
        continuous_operator — the continuous flux operator approximated
        __call__            — apply the operator: DiscreteField → DiscreteField
    """


__all__ = ["NumericalFlux"]
