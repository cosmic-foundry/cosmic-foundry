"""Manifold ABC."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from cosmic_foundry.continuous.topological_manifold import TopologicalManifold

if TYPE_CHECKING:
    from cosmic_foundry.continuous.atlas import Atlas


class Manifold(TopologicalManifold):
    """A topological manifold equipped with a smooth atlas.

    The atlas of charts constitutes the smooth structure: transition maps
    between overlapping charts are C∞ diffeomorphisms, making calculus
    possible on M.  The topological dimension ndim is inherited from
    TopologicalManifold.

    Subclasses add structure:
    - Manifold equipped with a metric — PseudoRiemannianManifold

    Required:
        atlas — the smooth atlas constituting the smooth structure of M
    """

    @property
    @abstractmethod
    def atlas(self) -> Atlas:
        """The smooth atlas constituting the smooth structure of M."""


__all__ = ["Manifold"]
