"""ManifoldWithBoundary ABC."""

from __future__ import annotations

from abc import abstractmethod

from cosmic_foundry.theory.manifold import Manifold


class ManifoldWithBoundary(Manifold):
    """A manifold whose boundary ∂M consists of manifolds of dimension ndim-1.

    Every interior point has a neighborhood homeomorphic to ℝⁿ; every
    boundary point has a neighborhood homeomorphic to the closed half-space
    ℝⁿ₊ = {x ∈ ℝⁿ : xₙ ≥ 0}.  Each boundary piece is a manifold of
    dimension ndim-1.

    boundary returns a tuple rather than a single object because practical
    simulation domains (e.g. boxes) have piecewise-smooth boundaries —
    multiple flat faces joined at edges — rather than a single smooth surface.

    Required:
        ndim     — topological dimension of this manifold (inherited)
        boundary — ∂M as a tuple of manifolds, each of dimension ndim-1
    """

    @property
    @abstractmethod
    def boundary(self) -> tuple[ManifoldWithBoundary, ...]:
        """∂M — boundary pieces, each a manifold of dimension ndim-1."""


__all__ = ["ManifoldWithBoundary"]
