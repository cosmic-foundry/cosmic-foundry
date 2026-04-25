"""DiffusionOperator: continuous operator -d: Ω⁰ → Ω¹ (negated gradient)."""

from __future__ import annotations

from typing import Any

import sympy

from cosmic_foundry.theory.continuous.differential_form import (
    DifferentialForm,
    OneForm,
    ZeroForm,
)
from cosmic_foundry.theory.continuous.differential_operator import DifferentialOperator
from cosmic_foundry.theory.continuous.manifold import Manifold


class _NegatedGradientField(OneForm[Any]):
    """The covector field -∇f for a scalar field f, as a concrete OneForm.

    Stores the negated gradient components as SymPy expressions derived
    by differentiating f.expr with respect to each symbol in f.symbols.
    component(i) returns the i-th component -∂f/∂xᵢ.
    """

    def __init__(self, field: DifferentialForm[Any, Any]) -> None:
        self._source = field
        self._components = tuple(-sympy.diff(field.expr, s) for s in field.symbols)

    @property
    def manifold(self) -> Any:
        return self._source.manifold

    @property
    def expr(self) -> sympy.Expr:
        return self._components[0]

    @property
    def symbols(self) -> tuple[sympy.Symbol, ...]:
        return self._source.symbols

    def component(self, i: int) -> sympy.Expr:
        """Return the i-th component -∂f/∂xᵢ."""
        return self._components[i]


class DiffusionOperator(DifferentialOperator[ZeroForm, OneForm]):
    """Continuous diffusion operator -d: Ω⁰ → Ω¹ (the negated gradient).

    DiffusionOperator represents the flux field F(φ) = -∇φ — the continuous
    operator that DiffusiveFlux approximates at convergence order p.  It is
    the Ω⁰ → Ω¹ leg of the commutation diagram

        φ ──────(-∇)──────▶ -∇φ
        │                       │
       (Rₕ, degree=n)   (Rₕ, degree=n-1)
        │                       │
        ▼                       ▼
        Uₕ ──(DiffusiveFlux)──▶ Fₕ

    DiffusionOperator earns its class as a concrete DifferentialOperator:
    both type parameters are fixed (ZeroForm → OneForm) and manifold is
    supplied at construction time.

    Parameters
    ----------
    manifold:
        The manifold on which the operator acts.
    """

    def __init__(self, manifold: Manifold) -> None:
        self._manifold = manifold

    @property
    def manifold(self) -> Manifold:
        return self._manifold

    @property
    def order(self) -> int:
        """Order of differentiation; 1 for the first-order gradient."""
        return 1

    def __call__(self, phi: ZeroForm) -> OneForm:
        """Apply the operator: φ ↦ -∇φ."""
        return _NegatedGradientField(phi)


__all__ = ["DiffusionOperator"]
