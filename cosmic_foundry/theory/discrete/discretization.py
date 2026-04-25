"""Discretization ABC."""

from __future__ import annotations

from abc import abstractmethod

from cosmic_foundry.theory.continuous.divergence_form_equation import (
    DivergenceFormEquation,
)
from cosmic_foundry.theory.discrete.discrete_operator import DiscreteOperator
from cosmic_foundry.theory.discrete.mesh import Mesh
from cosmic_foundry.theory.foundation.numeric_function import NumericFunction


class Discretization(NumericFunction[DivergenceFormEquation, DiscreteOperator]):
    """Maps a DivergenceFormEquation to a DiscreteOperator at a given mesh.

    A Discretization encapsulates the scheme choice — reconstruction,
    numerical flux, quadrature — for a particular mesh and approximation
    order.  Calling it with a DivergenceFormEquation produces the
    DiscreteOperator Lₕ that makes the commutation diagram

        Lₕ ∘ Rₕ ≈ Rₕ ∘ L   (up to O(hᵖ))

    hold, where p is the approximation order.  Different scheme choices
    are different ways of constructing Lₕ to satisfy this diagram at order p.

    For Lanes B and C, the commutation diagram must be verified via the
    convergence oracle framework in tests/support/.  The approximation order
    is a property of the concrete scheme — proved by the convergence test —
    not a parameter of the abstract interface.

    Required:
        mesh     — the mesh on which the scheme is defined
        __call__ — produce the DiscreteOperator for a given DivergenceFormEquation
                   (inherited from NumericFunction)
    """

    @property
    @abstractmethod
    def mesh(self) -> Mesh:
        """The mesh on which the scheme is defined."""


__all__ = ["Discretization"]
