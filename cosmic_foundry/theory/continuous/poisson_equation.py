"""PoissonEquation: ABC for -∇²φ = ρ in divergence form."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import sympy

from cosmic_foundry.theory.continuous.divergence_form_equation import (
    DivergenceFormEquation,
)
from cosmic_foundry.theory.continuous.field import Field, TensorField
from cosmic_foundry.theory.continuous.manifold import Manifold
from cosmic_foundry.theory.foundation.function import Function
from cosmic_foundry.theory.foundation.numeric_function import NumericFunction


class _NegatedGradientField(TensorField[Any, sympy.Expr]):
    """The (0,1)-covector field -∇f for a scalar field f.

    Stores the negated gradient components as SymPy expressions derived
    by differentiating f.expr with respect to each symbol in f.symbols.
    component(i) returns the i-th component -∂f/∂xᵢ.
    """

    def __init__(self, field: Field) -> None:
        self._source = field
        self._components = tuple(-sympy.diff(field.expr, s) for s in field.symbols)

    @property
    def manifold(self) -> Manifold:
        return self._source.manifold

    @property
    def tensor_type(self) -> tuple[int, int]:
        return (0, 1)

    @property
    def expr(self) -> sympy.Expr:
        # No canonical single scalar expression for a covector field; return
        # the first component to satisfy the SymbolicFunction protocol.
        return self._components[0]

    @property
    def symbols(self) -> tuple[sympy.Symbol, ...]:
        return self._source.symbols

    def component(self, i: int) -> sympy.Expr:
        """Return the i-th component -∂f/∂xᵢ as a SymPy expression."""
        return self._components[i]


class _NegatedGradientFlux(NumericFunction[Field, TensorField]):
    """The flux function F(φ) = -∇φ for the Poisson equation.

    Evaluates symbolically: given a scalar Field, returns a
    _NegatedGradientField whose i-th component is -∂φ/∂xᵢ.
    """

    def __call__(self, field: Field, **_: Any) -> _NegatedGradientField:
        return _NegatedGradientField(field)


class _ScalarField(Field[Any, sympy.Expr]):
    """A scalar field defined by a SymPy expression."""

    def __init__(
        self,
        manifold: Manifold,
        expr: sympy.Expr,
        symbols: tuple[sympy.Symbol, ...],
    ) -> None:
        self._manifold = manifold
        self._expr = expr
        self._symbols = symbols

    @property
    def manifold(self) -> Manifold:
        return self._manifold

    @property
    def expr(self) -> sympy.Expr:
        return self._expr

    @property
    def symbols(self) -> tuple[sympy.Symbol, ...]:
        return self._symbols


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
        source   — the right-hand side field ρ

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
    def source(self) -> Field:
        """The right-hand side field ρ in -∇²φ = ρ."""

    @property
    def flux(self) -> Function[Field, TensorField]:
        """Derived: the negated gradient flux F(φ) = -∇φ."""
        return _NEGATED_GRADIENT_FLUX

    def __call__(self, field: Field, **_: Any) -> _ScalarField:
        """Apply the Poisson operator L(φ) = -∇²φ.

        Computes -Σᵢ ∂²φ/∂xᵢ² symbolically via SymPy and returns it
        as a scalar Field on field.manifold.  The symbols are taken from
        field.symbols; the result is the strong-form residual that equals
        source.expr when φ solves the equation.
        """
        neg_laplacian = -sum(sympy.diff(field.expr, s, 2) for s in field.symbols)
        return _ScalarField(field.manifold, neg_laplacian, field.symbols)


__all__ = ["PoissonEquation"]
