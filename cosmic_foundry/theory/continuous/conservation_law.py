"""ConservationLaw: hyperbolic conservation law ∂ₜU + ∇·F(U) = S."""

from __future__ import annotations

from cosmic_foundry.theory.continuous.divergence_form_equation import (
    DivergenceFormEquation,
)


class ConservationLaw(DivergenceFormEquation):
    """A hyperbolic conservation law: ∂ₜU + ∇·F(U) = S, with F algebraic in U.

    ConservationLaw narrows DivergenceFormEquation to the case where the
    flux function F depends algebraically on U — not on its derivatives.
    This is the defining property of hyperbolic systems (Euler equations,
    scalar advection, MHD): the principal symbol of the linearised spatial
    operator has real eigenvalues, and the equation admits characteristics.

    The ∂ₜ term is handled by the time integrator (Epoch 2), not this object.
    This separation is preserved under the 3+1 ADM decomposition in GR:
    the covariant form ∇_μ F^μ = S decomposes to a spatial divergence with
    metric factors entering through the Chart and source.

    Required:
        flux     — algebraic flux function F: Field → TensorField (inherited)
        source   — source field S (inherited)
        manifold — the manifold on which this operator acts (inherited)

    Derived:
        order    — 1 (inherited from DivergenceFormEquation)
    """


__all__ = ["ConservationLaw"]
