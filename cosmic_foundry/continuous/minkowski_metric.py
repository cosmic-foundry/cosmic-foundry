"""MinkowskiMetric: the flat Lorentzian metric g = diag(+1, -1, -1, -1)."""

from __future__ import annotations

from typing import Any

import sympy

from cosmic_foundry.continuous.metric_tensor import MetricTensor
from cosmic_foundry.continuous.pseudo_riemannian_manifold import (
    PseudoRiemannianManifold,
)


class MinkowskiMetric(MetricTensor):
    """The Minkowski metric in the (+, −, −, −) sign convention.

    Components in standard inertial coordinates are diag(+1, −1, −1, −1),
    independent of position.  This is the unique flat Lorentzian metric on
    ℝ¹˒³ in these coordinates.
    """

    def __init__(self, manifold: PseudoRiemannianManifold) -> None:
        self._manifold = manifold

    @property
    def manifold(self) -> PseudoRiemannianManifold:
        """The pseudo-Riemannian manifold this metric is defined on."""
        return self._manifold

    def __call__(self, *args: Any, **kwargs: Any) -> sympy.Matrix:
        """Return the metric components: diag(+1, −1, −1, −1)."""
        return sympy.diag(1, -1, -1, -1)


__all__ = ["MinkowskiMetric"]
