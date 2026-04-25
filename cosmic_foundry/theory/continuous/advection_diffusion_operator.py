"""AdvectionDiffusionOperator: continuous operator F(φ) = φ − κ∇φ: Ω⁰ → Ω¹."""

from __future__ import annotations

from typing import Any

import sympy

from cosmic_foundry.theory.continuous.differential_form import OneForm, ZeroForm
from cosmic_foundry.theory.continuous.differential_operator import DifferentialOperator
from cosmic_foundry.theory.continuous.manifold import Manifold


class _AdvectionDiffusionFluxField(OneForm[Any]):
    """The covector field φ − κ∇φ for scalar field φ and diffusivity κ."""

    def __init__(self, field: ZeroForm[Any], kappa: sympy.Expr) -> None:
        super().__init__(
            field.manifold,
            tuple(
                field.expr - kappa * sympy.diff(field.expr, s) for s in field.symbols
            ),
            field.symbols,
        )


class AdvectionDiffusionOperator(DifferentialOperator[ZeroForm, OneForm]):
    """Continuous advection-diffusion operator F(φ) = φ − κ∇φ: Ω⁰ → Ω¹.

    AdvectionDiffusionOperator represents the combined flux field that
    AdvectionDiffusionFlux approximates at convergence order p.  It is
    the Ω⁰ → Ω¹ leg of the commutation diagram

        φ ─────(φ − κ∇φ)──────▶ φ − κ∇φ
        │                              │
       (Rₕ, degree=n)        (Rₕ, degree=n-1)
        │                              │
        ▼                              ▼
        Uₕ ──(AdvectionDiffusionFlux)──▶ Fₕ

    The divergence ∇·F = ∇·φ − κ∆φ combines a first-order advective term
    (unit velocity) with a second-order diffusive term weighted by κ.  The
    operator has order 1 because it differentiates φ once through the κ∇φ
    component.

    Parameters
    ----------
    manifold:
        The manifold on which the operator acts.
    kappa:
        Diffusion coefficient (default: 1).
    """

    def __init__(
        self,
        manifold: Manifold,
        kappa: sympy.Expr | None = None,
    ) -> None:
        self._manifold = manifold
        self._kappa: sympy.Expr = sympy.Integer(1) if kappa is None else kappa

    @property
    def manifold(self) -> Manifold:
        return self._manifold

    @property
    def order(self) -> int:
        """Order of differentiation; 1 for the first-order gradient in −κ∇φ."""
        return 1

    def __call__(self, phi: ZeroForm) -> OneForm:
        """Apply the operator: φ ↦ φ − κ∇φ."""
        return _AdvectionDiffusionFluxField(phi, self._kappa)


__all__ = ["AdvectionDiffusionOperator"]
