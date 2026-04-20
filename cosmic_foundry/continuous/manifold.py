"""Manifold ABC."""

from __future__ import annotations

from abc import abstractmethod

from cosmic_foundry.foundation.set import Set


class Manifold(Set):
    """A topological space that locally resembles ℝⁿ.

    Every point has a neighborhood homeomorphic to an open subset of ℝⁿ
    (or the half-space ℝⁿ₊ for manifolds with boundary).  The integer n
    is the topological dimension, shared by all points.

    Subclasses add structure:
    - SmoothManifold — C∞ transition maps; calculus is possible

    Required:
        ndim — topological dimension of this manifold
    """

    @property
    @abstractmethod
    def ndim(self) -> int:
        """Topological dimension of this manifold."""


__all__ = ["Manifold"]
