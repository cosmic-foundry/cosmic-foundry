"""ManifoldWithBoundary ABC."""

from __future__ import annotations

from abc import abstractmethod

from cosmic_foundry.theory.manifold import Manifold
from cosmic_foundry.theory.smooth_manifold import SmoothManifold


class ManifoldWithBoundary(Manifold):
    """A manifold whose boundary ∂M consists of smooth manifolds of dimension ndim-1.

    Every interior point has a neighborhood homeomorphic to ℝⁿ; every
    boundary point has a neighborhood homeomorphic to the closed half-space
    ℝⁿ₊ = {x ∈ ℝⁿ : xₙ ≥ 0}.  Each boundary piece is a smooth manifold
    (without boundary — ∂∂M = ∅) of dimension ndim-1.

    boundary returns a tuple rather than a single SmoothManifold because
    practical simulation domains (e.g. boxes) have piecewise-smooth boundaries
    — multiple flat faces joined at edges — rather than a single smooth surface.

    Required:
        ndim     — topological dimension of this manifold (inherited)
        boundary — ∂M as a tuple of SmoothManifolds, each of dimension ndim-1
    """

    @property
    @abstractmethod
    def boundary(self) -> tuple[SmoothManifold, ...]:
        """∂M — boundary pieces, each a SmoothManifold of dimension ndim-1."""


__all__ = ["ManifoldWithBoundary"]
