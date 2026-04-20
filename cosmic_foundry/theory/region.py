"""Region: a compact, connected simulation domain."""

from __future__ import annotations

from abc import abstractmethod

from cosmic_foundry.theory.manifold_with_boundary import ManifoldWithBoundary
from cosmic_foundry.theory.smooth_manifold import SmoothManifold


class Region(ManifoldWithBoundary):  # noqa: B024
    """A compact, connected region Ω of a smooth manifold M.

    A Region is a ManifoldWithBoundary that is:
      - compact  — bounded and closed; enables integration over Ω
      - connected — a single piece; no disjoint sub-domains

    The region lives inside an ambient smooth manifold M; its dimension
    equals that of M, and its boundary ∂Ω is a ManifoldWithBoundary of
    dimension ndim - 1.

    ndim is derived from the ambient manifold so that concrete subclasses
    need only declare ambient_manifold.

    Required:
        ambient_manifold — the smooth manifold M that contains Ω
        boundary         — the faces ∂Ω (inherited abstract from
                           ManifoldWithBoundary)
    """

    @property
    @abstractmethod
    def ambient_manifold(self) -> SmoothManifold:
        """The smooth manifold M in which this region is embedded."""

    @property
    def ndim(self) -> int:
        return self.ambient_manifold.ndim


__all__ = ["Region"]
