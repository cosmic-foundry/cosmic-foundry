"""EuclideanSpace — flat positive-definite ℝⁿ."""

from __future__ import annotations

from cosmic_foundry.continuous.euclidean_metric import EuclideanMetric
from cosmic_foundry.continuous.identity_chart import IdentityChart
from cosmic_foundry.continuous.riemannian_manifold import RiemannianManifold
from cosmic_foundry.continuous.single_chart_atlas import SingleChartAtlas


class EuclideanSpace(RiemannianManifold):
    """ℝⁿ with the standard flat positive-definite metric.

    The only free parameter is the dimension n.  signature = (n, 0) and
    ndim = n are both derived — signature from RiemannianManifold, ndim
    from this class.

    Required:
        n — dimension of the Euclidean space
    """

    def __init__(self, n: int) -> None:
        if n < 1:
            raise ValueError(f"dimension must be at least 1, got {n}")
        self._n = n

    @property
    def ndim(self) -> int:
        """Dimension of this Euclidean space."""
        return self._n

    @property
    def metric(self) -> EuclideanMetric:
        """The flat Euclidean metric: g_ij = δ_ij."""
        return EuclideanMetric(self)

    @property
    def atlas(self) -> SingleChartAtlas:
        """The standard atlas: one global identity chart covering all of ℝⁿ."""
        return SingleChartAtlas(IdentityChart(self))


__all__ = [
    "EuclideanSpace",
]
