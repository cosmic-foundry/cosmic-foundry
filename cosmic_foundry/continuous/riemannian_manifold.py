"""RiemannianManifold ABC."""

from __future__ import annotations

from abc import abstractmethod

from cosmic_foundry.continuous.pseudo_riemannian_manifold import (
    PseudoRiemannianManifold,
)


class RiemannianManifold(PseudoRiemannianManifold):
    """A PseudoRiemannianManifold whose metric is positive-definite.

    A Riemannian manifold (M, g) specializes the pseudo-Riemannian case to
    signature (ndim, 0): all eigenvalues of g are strictly positive.  This
    makes g a true inner product at each point, giving rise to lengths,
    angles, geodesic distance, and volume forms in the familiar Euclidean
    sense.

    Newtonian spatial domains (flat ℝⁿ with Euclidean metric, spheres,
    tori, etc.) are Riemannian manifolds.  Lorentzian spacetimes are not —
    they live at PseudoRiemannianManifold.

    The positive-definiteness constraint (q = 0) is enforced by deriving
    signature from ndim.  Concrete subclasses need only declare ndim;
    signature is not a free parameter at this level.

    Required:
        ndim — dimension of the manifold; signature = (ndim, 0) follows
    """

    @property
    @abstractmethod
    def ndim(self) -> int:
        """Dimension of this Riemannian manifold."""

    @property
    def signature(self) -> tuple[int, int]:
        """Metric signature, derived as (ndim, 0)."""
        return (self.ndim, 0)


__all__ = [
    "RiemannianManifold",
]
