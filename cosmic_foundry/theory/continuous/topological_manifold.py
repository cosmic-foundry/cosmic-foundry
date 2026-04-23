"""TopologicalManifold ABC."""

from __future__ import annotations

from abc import abstractmethod

from cosmic_foundry.theory.foundation.topological_space import TopologicalSpace


class TopologicalManifold(TopologicalSpace):
    """A topological space that is locally homeomorphic to ℝⁿ.

    Every point p ∈ M has an open neighborhood U ∋ p and a homeomorphism
    φ: U → V onto an open subset V ⊆ ℝⁿ.  The integer n is the topological
    dimension of M, uniform across all points.

    This class encodes the purely topological layer of manifold structure:
    local Euclidean character and dimension.  It carries no smooth structure
    and no metric — those are added by subclasses.

    Subclasses add structure:
    - Topological manifold equipped with a smooth atlas — Manifold

    Required:
        ndim — topological dimension of this manifold
    """

    @property
    @abstractmethod
    def ndim(self) -> int:
        """Topological dimension n such that every point has a neighborhood
        homeomorphic to an open subset of ℝⁿ."""


__all__ = ["TopologicalManifold"]
