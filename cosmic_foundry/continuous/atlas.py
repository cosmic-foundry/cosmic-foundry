"""Atlas ABC: a collection of charts constituting a smooth structure."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from cosmic_foundry.foundation.indexed_family import IndexedFamily

if TYPE_CHECKING:
    from cosmic_foundry.continuous.chart import Chart
    from cosmic_foundry.continuous.smooth_manifold import SmoothManifold


class Atlas(IndexedFamily):
    """A collection of charts whose domains cover a smooth manifold M.

    A smooth manifold is defined as the pair (M, A) — a topological manifold
    together with a maximal smooth atlas.  The atlas is not a property of M;
    it is constitutive of M's smooth structure.  Transition maps φβ ∘ φα⁻¹
    between overlapping charts are required to be C∞ diffeomorphisms.

    An Atlas is an IndexedFamily of Charts: charts are retrievable by integer
    index, and the union of their domains covers all of M.

    Required:
        manifold     — the smooth manifold this atlas covers
        __getitem__  — retrieve the chart at index i (narrows to Chart)
        __len__      — number of charts in this atlas
    """

    @property
    @abstractmethod
    def manifold(self) -> SmoothManifold:
        """The smooth manifold M that this atlas covers."""

    @abstractmethod
    def __getitem__(self, index: int) -> Chart:
        """Return the chart at *index*."""

    @abstractmethod
    def __len__(self) -> int:
        """Number of charts in this atlas."""


__all__ = ["Atlas"]
