"""PoissonEquation: ABC for -∇²φ = ρ in divergence form."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import sympy

from cosmic_foundry.theory.continuous.differential_form import (
    DifferentialForm,
    OneForm,
    ZeroForm,
)
from cosmic_foundry.theory.continuous.diffusion_operator import _NegatedGradientField
from cosmic_foundry.theory.continuous.divergence_form_equation import (
    DivergenceFormEquation,
)
from cosmic_foundry.theory.continuous.manifold import Manifold
from cosmic_foundry.theory.foundation.function import Function
from cosmic_foundry.theory.foundation.numeric_function import NumericFunction


class _NegatedGradientFlux(NumericFunction[DifferentialForm, OneForm]):
    """The flux function F(φ) = -∇φ for the Poisson equation.

    Evaluates symbolically: given a DifferentialForm, returns a
    _NegatedGradientField whose i-th component is -∂φ/∂xᵢ.
    """

    def __call__(self, field: DifferentialForm[Any, Any]) -> OneForm:
        return _NegatedGradientField(field)


_NEGATED_GRADIENT_FLUX = _NegatedGradientFlux()


class PoissonEquation(DivergenceFormEquation):
    """Abstract base for the Poisson equation: -∇²φ = ρ.

    PoissonEquation is the divergence-form statement

        ∇·(-∇φ) = ρ,

    obtained from -∇²φ = ρ by recognizing the left-hand side as the
    divergence of the flux F(φ) = -∇φ.  The sign convention (flux = -∇φ,
    not +∇φ) ensures the discrete operator assembled by FVMDiscretization
    is positive definite — see C4 and C5.

    There is no LaplaceOperator class: -∇²φ = -∇·∇φ is fully captured by
    the flux field -∇φ and the divergence theorem; it does not earn a class
    under the falsifiable-constraint rule.

    PoissonEquation earns its class by deriving flux = -∇(·): the flux
    degree of freedom present in DivergenceFormEquation is removed.
    Concrete subclasses supply manifold and source.

    Required:
        manifold — the domain on which the equation is posed
        source   — the right-hand side scalar field ρ

    Derived:
        flux     — the negated gradient operator F(φ) = -∇φ
        order    — 1 (inherited from DivergenceFormEquation)
    """

    @property
    @abstractmethod
    def manifold(self) -> Manifold:
        """The manifold on which this equation is posed."""

    @property
    @abstractmethod
    def source(self) -> ZeroForm:
        """The right-hand side scalar field ρ in -∇²φ = ρ."""

    @property
    def flux(self) -> Function[DifferentialForm, OneForm]:
        """Derived: the negated gradient flux F(φ) = -∇φ."""
        return _NEGATED_GRADIENT_FLUX

    def __call__(self, field: DifferentialForm) -> ZeroForm:
        """Apply the Poisson operator L(φ) = -∇²φ.

        Computes -Σᵢ ∂²φ/∂xᵢ² symbolically via SymPy and returns it
        as a scalar ZeroForm on field.manifold.  The symbols are taken from
        field.symbols; the result is the strong-form residual that equals
        source.expr when φ solves the equation.
        """
        neg_laplacian = -sum(sympy.diff(field.expr, s, 2) for s in field.symbols)
        return ZeroForm(field.manifold, neg_laplacian, field.symbols)


__all__ = ["PoissonEquation"]
