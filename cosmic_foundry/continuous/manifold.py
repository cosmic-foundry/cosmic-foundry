"""Manifold ABCs: topological structure and smooth atlas."""

from __future__ import annotations

from abc import abstractmethod

from cosmic_foundry.continuous.topological_manifold import TopologicalManifold
from cosmic_foundry.foundation.homeomorphism import Homeomorphism
from cosmic_foundry.foundation.indexed_family import IndexedFamily


class Atlas(IndexedFamily):
    """A family of homeomorphisms constituting the smooth structure of a manifold.

    An atlas gives a manifold its smooth structure: transition maps φβ ∘ φα⁻¹
    between overlapping charts are required to be C∞ diffeomorphisms.

    An Atlas is an IndexedFamily of Homeomorphisms: charts are retrievable by
    integer index, and the union of their domains covers all of M.

    Required:
        __getitem__  — retrieve the homeomorphism at index i
        __len__      — number of homeomorphisms in this atlas
    """

    @abstractmethod
    def __getitem__(self, index: int) -> Homeomorphism:
        """Return the homeomorphism at *index*."""

    @abstractmethod
    def __len__(self) -> int:
        """Number of homeomorphisms in this atlas."""


class Manifold(TopologicalManifold):
    """A topological manifold equipped with a smooth atlas.

    The atlas of homeomorphisms constitutes the smooth structure: transition
    maps between overlapping charts are C∞ diffeomorphisms, making calculus
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


__all__ = ["Atlas", "Manifold"]
