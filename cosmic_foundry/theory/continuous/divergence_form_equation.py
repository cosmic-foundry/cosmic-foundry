"""DivergenceFormEquation: spatial PDE in divergence form ∇·F(U) = S."""

from __future__ import annotations

from abc import abstractmethod

from cosmic_foundry.theory.continuous.differential_form import (
    DifferentialForm,
    OneForm,
    ZeroForm,
)
from cosmic_foundry.theory.continuous.differential_operator import DifferentialOperator
from cosmic_foundry.theory.foundation.function import Function


class DivergenceFormEquation(DifferentialOperator[DifferentialForm, ZeroForm]):
    """A PDE expressed as a spatial divergence: ∇·F(U) = S.

    DivergenceFormEquation is the formal bridge between a continuous PDE
    and its finite-volume discretization.  The integral form

        ∮_∂Ωᵢ F(U)·n̂ dA = ∫_Ωᵢ S dV

    is obtained by integrating over any control volume Ωᵢ and applying
    the divergence theorem.  This integral form — not the differential form
    — is what DivergenceFormDiscretization assembles: the NumericalFlux evaluates the
    left-hand side face by face.

    The operator is spatial only.  Time derivatives are handled by the time
    integrator (Epoch 2), not by this object.

    The domain is DifferentialForm (open: any degree, pending decision 3
    sub-question (a) on multi-component input); the codomain is ZeroForm
    because ∇·F is always a scalar.

    Required:
        flux     — the flux function F: DifferentialForm → OneForm
        source   — the source scalar field S
        manifold — the manifold on which this operator acts (inherited)

    Derived:
        order    — 1; a first-order divergence operator
    """

    @property
    @abstractmethod
    def flux(self) -> Function[DifferentialForm, OneForm]:
        """The flux function F in ∇·F(U) = S."""

    @property
    @abstractmethod
    def source(self) -> ZeroForm:
        """The source scalar field S in ∇·F(U) = S."""

    @property
    def order(self) -> int:
        """Differentiation order; derived as 1 for a first-order divergence operator."""
        return 1


__all__ = ["DivergenceFormEquation"]
