"""Discretization ABC."""

from __future__ import annotations

from abc import abstractmethod

from cosmic_foundry.theory.continuous.conservation_law import ConservationLaw
from cosmic_foundry.theory.discrete.discrete_operator import DiscreteOperator
from cosmic_foundry.theory.discrete.mesh import Mesh
from cosmic_foundry.theory.foundation.numeric_function import NumericFunction


class Discretization(NumericFunction[ConservationLaw, DiscreteOperator]):
    """Maps a ConservationLaw to a DiscreteOperator at a given mesh and order.

    A Discretization encapsulates the scheme choice — reconstruction,
    Riemann solver, quadrature — for a particular mesh and approximation
    order.  Calling it with a ConservationLaw produces the DiscreteOperator
    Lₕ that makes the commutation diagram

        Lₕ ∘ Rₕ ≈ Rₕ ∘ L   (up to O(hᵖ))

    hold, where p is the approximation order.  Different scheme choices
    are different ways of constructing Lₕ to satisfy this diagram at order p.

    For Lanes B and C, the commutation diagram must be verified algebraically
    via SymPy as an assert_* function in tests/, following the standard test
    authorship convention.

    Required:
        mesh     — the mesh on which the scheme is defined
        order    — the approximation order p
        __call__ — produce the DiscreteOperator for a given ConservationLaw
                   (inherited from NumericFunction)
    """

    @property
    @abstractmethod
    def mesh(self) -> Mesh:
        """The mesh on which the scheme is defined."""

    @property
    @abstractmethod
    def order(self) -> int:
        """The approximation order p."""


__all__ = ["Discretization"]
