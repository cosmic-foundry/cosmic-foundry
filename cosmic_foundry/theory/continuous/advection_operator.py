"""AdvectionOperator: continuous operator id⊗v: Ω⁰ → Ω¹ (scalar advection flux)."""

from __future__ import annotations

from typing import Any

import sympy

from cosmic_foundry.theory.continuous.differential_form import OneForm, ZeroForm
from cosmic_foundry.theory.continuous.differential_operator import DifferentialOperator
from cosmic_foundry.theory.continuous.manifold import Manifold


class _AdvectionFluxField(OneForm[Any]):
    """The covector field v·φ for scalar field φ and constant velocity v."""

    def __init__(self, field: ZeroForm[Any], velocity: sympy.Expr) -> None:
        super().__init__(
            field.manifold,
            tuple(velocity * field.expr for _ in field.symbols),
            field.symbols,
        )


class AdvectionOperator(DifferentialOperator[ZeroForm, OneForm]):
    """Continuous advection operator v·(·): Ω⁰ → Ω¹ (scalar flux field).

    AdvectionOperator represents the flux field F(φ) = v·φ — the continuous
    operator that AdvectiveFlux approximates at convergence order p.  It is
    the Ω⁰ → Ω¹ leg of the commutation diagram

        φ ─────(v·φ)──────▶ v·φ
        │                       │
       (Rₕ, degree=n)   (Rₕ, degree=n-1)
        │                       │
        ▼                       ▼
        Uₕ ──(AdvectiveFlux)──▶ Fₕ

    Parameters
    ----------
    manifold:
        The manifold on which the operator acts.
    velocity:
        Constant advection velocity (default: 1).
    """

    def __init__(
        self,
        manifold: Manifold,
        velocity: sympy.Expr | None = None,
    ) -> None:
        self._manifold = manifold
        self._velocity: sympy.Expr = sympy.Integer(1) if velocity is None else velocity

    @property
    def manifold(self) -> Manifold:
        return self._manifold

    @property
    def order(self) -> int:
        """Order of differentiation; 0 for pointwise multiplication."""
        return 0

    def __call__(self, phi: ZeroForm) -> OneForm:
        """Apply the operator: φ ↦ v·φ."""
        return _AdvectionFluxField(phi, self._velocity)


__all__ = ["AdvectionOperator"]
