"""DiscreteOperator ABC."""

from __future__ import annotations

from abc import abstractmethod
from typing import TypeVar

from cosmic_foundry.theory.discrete.mesh_function import MeshFunction
from cosmic_foundry.theory.foundation.numeric_function import NumericFunction

_V = TypeVar("_V")


class DiscreteOperator(NumericFunction[MeshFunction[_V], MeshFunction[_V]]):
    """The discrete analog of DifferentialOperator: Lₕ: MeshFunction → MeshFunction.

    A DiscreteOperator is the output of a Discretization — the Lₕ that makes
    the commutation diagram Lₕ ∘ Rₕ ≈ Rₕ ∘ L hold to the chosen approximation
    order.  It earns its class via the verifiable integer claim .order: int —
    the composite convergence order of the approximation scheme.

    In C4, DiscreteOperator gains a second required property:
    .continuous_operator: DifferentialOperator — the continuous operator this
    instance approximates.  Together order and continuous_operator are the two
    falsifiable claims every discrete operator makes.

    A DiscreteOperator is not constructed directly from stencil coefficients;
    it is produced by a Discretization, which threads continuous_operator
    automatically (it is the input L to the Discretization).

    Required:
        order    — composite convergence order of the approximation scheme
        __call__ — apply the operator (inherited from NumericFunction)
    """

    @property
    @abstractmethod
    def order(self) -> int:
        """Composite convergence order of the approximation scheme."""


__all__ = ["DiscreteOperator"]
