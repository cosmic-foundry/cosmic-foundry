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
    """The covector field -∇f for a scalar field f."""

    def __init__(self, field: DifferentialForm[Any, Any]) -> None:
        super().__init__(
            field.manifold,
            tuple(-sympy.diff(field.expr, s) for s in field.symbols),
            field.symbols,
        )


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
