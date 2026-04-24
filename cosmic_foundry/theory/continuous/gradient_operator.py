"""GradientOperator: abstract first-order differential operator ∇."""

from __future__ import annotations

from cosmic_foundry.theory.continuous.differential_operator import DifferentialOperator


class GradientOperator(DifferentialOperator):
    """Abstract gradient operator ∇: scalar Field → covariant TensorField.

    The gradient of a scalar field f on a manifold M is the (0,1)-type
    covector field ∇f with components (∂f/∂x¹, …, ∂f/∂xⁿ) in any local
    coordinate chart.  On a Riemannian manifold the index can be raised with
    the metric to give the contravariant gradient; concretely on a
    EuclideanManifold with the flat metric these coincide.

    GradientOperator earns its class via the derived property order = 1.
    Concrete subclasses implement __call__ for a specific manifold type and
    coordinate chart.

    Required:
        manifold — the manifold on which this operator acts (inherited)

    Derived:
        order    — 1; the gradient is a first-order differential operator
    """

    @property
    def order(self) -> int:
        """Differentiation order; derived as 1 for the gradient operator."""
        return 1


__all__ = ["GradientOperator"]
