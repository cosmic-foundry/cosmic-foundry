"""EuclideanSpace — flat positive-definite ℝⁿ."""

from __future__ import annotations

from cosmic_foundry.continuous.flat_manifold import FlatManifold
from cosmic_foundry.continuous.riemannian_manifold import RiemannianManifold


class EuclideanSpace(RiemannianManifold, FlatManifold):
    """ℝⁿ with the standard flat positive-definite metric.

    The only free parameter is the dimension n.  signature = (n, 0) and
    ndim = n are both derived — signature from RiemannianManifold, ndim
    from this class.

    MRO: EuclideanSpace → RiemannianManifold → FlatManifold →
         PseudoRiemannianManifold → SmoothManifold → Set

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


__all__ = [
    "EuclideanSpace",
]
