"""Manifold ABC."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from cosmic_foundry.foundation.set import Set

if TYPE_CHECKING:
    from cosmic_foundry.continuous.atlas import Atlas


class Manifold(Set):
    """A topological space that locally resembles ℝⁿ, equipped with an atlas.

    Every point has a neighborhood homeomorphic to an open subset of ℝⁿ.
    The integer n is the topological dimension, shared by all points.  The
    atlas of charts constitutes the smooth structure: transition maps between
    overlapping charts are C∞ diffeomorphisms, making calculus possible on M.

    Subclasses add structure:
    - Manifold equipped with a metric — PseudoRiemannianManifold

    Required:
        ndim  — topological dimension of this manifold
        atlas — the smooth atlas constituting the smooth structure of M
    """

    @property
    @abstractmethod
    def ndim(self) -> int:
        """Topological dimension of this manifold."""

    @property
    @abstractmethod
    def atlas(self) -> Atlas:
        """The smooth atlas constituting the smooth structure of M."""


__all__ = ["Manifold"]
