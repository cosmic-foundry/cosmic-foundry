"""ManifoldWithBoundary ABC."""

from __future__ import annotations

from abc import abstractmethod

from cosmic_foundry.theory.manifold import Manifold
from cosmic_foundry.theory.smooth_manifold import SmoothManifold


class ManifoldWithBoundary(Manifold):
    """A manifold whose boundary ∂M is a smooth manifold of dimension ndim-1.

    Every interior point has a neighborhood homeomorphic to ℝⁿ; every
    boundary point has a neighborhood homeomorphic to the closed half-space
    ℝⁿ₊ = {x ∈ ℝⁿ : xₙ ≥ 0}.  The boundary ∂M is itself a smooth manifold
    (without boundary — ∂∂M = ∅) of dimension ndim-1.

    Required:
        ndim     — topological dimension of this manifold (inherited)
        boundary — ∂M as a SmoothManifold of dimension ndim-1
    """

    @property
    @abstractmethod
    def boundary(self) -> SmoothManifold:
        """∂M — the boundary manifold of dimension ndim-1."""


__all__ = ["ManifoldWithBoundary"]
