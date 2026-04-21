"""IdentityChart: the standard global chart for flat manifolds."""

from __future__ import annotations

from typing import Any

from cosmic_foundry.continuous.chart import Chart
from cosmic_foundry.continuous.manifold import Manifold
from cosmic_foundry.foundation.function import Function


class IdentityChart(Chart[Any, Any]):
    """The identity coordinate chart φ: M → ℝⁿ, φ(p) = p.

    Appropriate for flat manifolds (EuclideanSpace, MinkowskiSpace) that
    admit a single global coordinate system.  The chart maps every point
    to its standard coordinates and is its own inverse.

    The codomain is ℝⁿ as a coordinate space (EuclideanSpace(ndim)),
    where n = domain.ndim.

    Required:
        manifold — the manifold M this chart covers globally
    """

    def __init__(self, manifold: Manifold) -> None:
        self._manifold = manifold

    @property
    def domain(self) -> Manifold:
        """The manifold M."""
        return self._manifold

    @property
    def codomain(self) -> Any:
        """ℝⁿ as a coordinate space, n = domain.ndim."""
        from cosmic_foundry.continuous.euclidean_space import EuclideanSpace

        return EuclideanSpace(self._manifold.ndim)

    @property
    def inverse(self) -> Function[Any, Any]:
        """The identity is its own inverse."""
        return self

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Return the argument unchanged: the identity map."""
        return args[0] if args else kwargs


__all__ = ["IdentityChart"]
