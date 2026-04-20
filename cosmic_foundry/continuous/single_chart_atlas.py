"""SingleChartAtlas: an atlas consisting of one global chart."""

from __future__ import annotations

from cosmic_foundry.continuous.atlas import Atlas
from cosmic_foundry.continuous.chart import Chart
from cosmic_foundry.continuous.smooth_manifold import SmoothManifold


class SingleChartAtlas(Atlas):
    """An atlas consisting of exactly one chart covering all of M.

    Appropriate for manifolds that admit a single global coordinate system:
    EuclideanSpace ℝⁿ, MinkowskiSpace ℝ¹˒³, and any other flat manifold
    with a globally-defined coordinate chart.

    Required:
        chart — the single Chart whose domain is all of M
    """

    def __init__(self, chart: Chart) -> None:
        self._chart = chart

    @property
    def manifold(self) -> SmoothManifold:
        """The manifold covered by the single chart."""
        return self._chart.domain

    def __getitem__(self, index: int) -> Chart:
        if index != 0:
            raise IndexError(
                f"SingleChartAtlas has one chart; index {index} is out of range"
            )
        return self._chart

    def __len__(self) -> int:
        return 1


__all__ = ["SingleChartAtlas"]
