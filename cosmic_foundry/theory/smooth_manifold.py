"""SmoothManifold ABC."""

from __future__ import annotations

from abc import abstractmethod

from cosmic_foundry.theory.manifold import Manifold


class SmoothManifold(Manifold):
    """A Manifold equipped with a smooth (C∞) structure.

    A smooth manifold M is a topological space that locally looks like ℝⁿ,
    with smooth transition maps between overlapping coordinate charts.  The
    integer n is the dimension of M.

    Smooth structure is what makes calculus possible on M: it gives meaning
    to smooth functions, tangent vectors, differential forms, and eventually
    to tensor fields and metrics.  It does not by itself define distances or
    angles — those require a metric (see PseudoRiemannianManifold).

    Concrete subclasses represent specific geometric spaces: flat Euclidean
    ℝⁿ, spheres, tori, hyperbolic spaces, Lorentzian spacetimes, etc.

    Required:
        ndim — the topological dimension of M
    """

    @property
    @abstractmethod
    def ndim(self) -> int:
        """Topological dimension of this manifold."""


__all__ = [
    "SmoothManifold",
]
