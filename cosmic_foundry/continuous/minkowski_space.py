"""MinkowskiSpace — flat Lorentzian ℝ⁴."""

from __future__ import annotations

from cosmic_foundry.continuous.identity_chart import IdentityChart
from cosmic_foundry.continuous.minkowski_metric import MinkowskiMetric
from cosmic_foundry.continuous.pseudo_riemannian_manifold import (
    PseudoRiemannianManifold,
)
from cosmic_foundry.continuous.single_chart_atlas import SingleChartAtlas


class MinkowskiSpace(PseudoRiemannianManifold):
    """ℝ⁴ with Lorentzian signature (1, 3).

    The flat pseudo-Riemannian background for special-relativistic
    simulations.  No free parameters — dimension and signature are fixed.

    Convention: signature (1, 3) means one positive (time) and three
    negative (space) eigenvalues, consistent with the particle-physics
    sign convention (+, −, −, −).
    """

    @property
    def signature(self) -> tuple[int, int]:
        """Lorentzian signature: one timelike, three spacelike directions."""
        return (1, 3)

    @property
    def metric(self) -> MinkowskiMetric:
        """The flat Lorentzian metric: g = diag(+1, −1, −1, −1)."""
        return MinkowskiMetric(self)

    @property
    def atlas(self) -> SingleChartAtlas:
        """The standard atlas: one global chart covering all of ℝ¹˒³."""
        return SingleChartAtlas(IdentityChart(self))


__all__ = [
    "MinkowskiSpace",
]
