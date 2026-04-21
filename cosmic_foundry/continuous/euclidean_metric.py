"""EuclideanMetric: the flat Riemannian metric g_ij = δ_ij."""

from __future__ import annotations

from typing import Any

import sympy

from cosmic_foundry.continuous.metric_tensor import MetricTensor
from cosmic_foundry.continuous.riemannian_manifold import RiemannianManifold


class EuclideanMetric(MetricTensor):
    """The standard flat Riemannian metric: g_ij = δ_ij.

    In Cartesian coordinates the components are the n×n identity matrix,
    independent of position.  This is the unique flat, positive-definite
    metric on ℝⁿ in standard coordinates.
    """

    def __init__(self, manifold: RiemannianManifold) -> None:
        self._manifold = manifold

    @property
    def manifold(self) -> RiemannianManifold:
        """The Riemannian manifold this metric is defined on."""
        return self._manifold

    def __call__(self, *args: Any, **kwargs: Any) -> sympy.Matrix:
        """Return the metric components: the n×n identity matrix."""
        return sympy.eye(self._manifold.ndim)


__all__ = ["EuclideanMetric"]
