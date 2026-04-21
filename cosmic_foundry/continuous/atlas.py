"""Atlas ABC: a collection of charts constituting a smooth structure."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from cosmic_foundry.continuous.topological_manifold import TopologicalManifold
from cosmic_foundry.foundation.indexed_family import IndexedFamily

if TYPE_CHECKING:
    from cosmic_foundry.continuous.chart import Chart


class Atlas(IndexedFamily):
    """A collection of charts whose domains cover a topological manifold M.

    An atlas gives M its smooth structure: transition maps φβ ∘ φα⁻¹
    between overlapping charts are required to be C∞ diffeomorphisms.
    The atlas covers a TopologicalManifold — the pre-smooth object to
    which the atlas is being attached — so there is no import cycle with
    Manifold (which is TopologicalManifold + Atlas).

    An Atlas is an IndexedFamily of Charts: charts are retrievable by integer
    index, and the union of their domains covers all of M.

    Required:
        manifold     — the topological manifold this atlas covers
        __getitem__  — retrieve the chart at index i (narrows to Chart)
        __len__      — number of charts in this atlas
    """

    @property
    @abstractmethod
    def manifold(self) -> TopologicalManifold:
        """The topological manifold M that this atlas covers."""

    @abstractmethod
    def __getitem__(self, index: int) -> Chart:
        """Return the chart at *index*."""

    @abstractmethod
    def __len__(self) -> int:
        """Number of charts in this atlas."""


__all__ = ["Atlas"]
