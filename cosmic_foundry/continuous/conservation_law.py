"""ConservationLaw ABC."""

from __future__ import annotations

from abc import abstractmethod

from cosmic_foundry.continuous.differential_operator import DifferentialOperator
from cosmic_foundry.continuous.field import Field, TensorField
from cosmic_foundry.foundation.function import Function


class ConservationLaw(DifferentialOperator):
    """A differential operator in divergence form: ∇·F(U) in ∂ₜU + ∇·F(U) = S.

    A conservation law is a first-order differential operator whose action
    on a field U is the divergence of a flux function F(U), plus a source
    field S.  The divergence theorem converts this to integral form per
    cell: ∮_∂Ωᵢ F(U)·n dA = ∫_Ωᵢ S dV, which is the starting point for
    finite volume discretisation.

    The operator is spatial only: ∂ₜ is handled by the time integrator,
    not this object.  This separation is preserved under the 3+1 ADM
    decomposition in GR.

    Required:
        flux     — the flux function F mapping a Field to a TensorField
        source   — the source field S
        manifold — the manifold on which this operator acts (inherited)

    Derived:
        order    — differentiation order; derived as 1 (first-order divergence)
    """

    @property
    @abstractmethod
    def flux(self) -> Function[Field, TensorField]:
        """The flux function F in ∇·F(U) = S."""

    @property
    @abstractmethod
    def source(self) -> Field:
        """The source field S in ∇·F(U) = S."""

    @property
    def order(self) -> int:
        """Differentiation order; derived as 1 for a first-order divergence operator."""
        return 1


__all__ = ["ConservationLaw"]
