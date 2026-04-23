"""Manifold ABCs: diffeomorphism, chart, atlas, and smooth structure."""

from __future__ import annotations

from abc import abstractmethod
from typing import Generic, TypeVar

import sympy

from cosmic_foundry.theory.continuous.topological_manifold import TopologicalManifold
from cosmic_foundry.theory.foundation.homeomorphism import Homeomorphism
from cosmic_foundry.theory.foundation.indexed_family import IndexedFamily

D = TypeVar("D")
C = TypeVar("C")


class Diffeomorphism(Homeomorphism[D, C], Generic[D, C]):
    """A smooth bijection between manifolds with a smooth inverse: φ: U → V.

    A diffeomorphism is a homeomorphism whose forward map and inverse are
    both C∞. Diffeomorphisms are the isomorphisms of smooth manifolds —
    two manifolds connected by a diffeomorphism are smoothly identical.

    Narrows domain and codomain to Manifold, reflecting that smooth
    structure is the relevant context.

    Required:
        domain   — the source smooth manifold U
        codomain — the target smooth manifold V
        inverse  — the smooth inverse φ⁻¹: V → U
    """

    @property
    @abstractmethod
    def domain(self) -> Manifold:
        """The source smooth manifold U."""

    @property
    @abstractmethod
    def codomain(self) -> Manifold:
        """The target smooth manifold V."""

    @property
    @abstractmethod
    def inverse(self) -> Diffeomorphism[C, D]:
        """The smooth inverse φ⁻¹: V → U."""


class Chart(Diffeomorphism[D, C], Generic[D, C]):  # noqa: B024
    """A local coordinate system on a smooth manifold: φ: U → V.

    A chart maps an open subset U of a manifold M diffeomorphically onto an
    open subset V of ℝⁿ. The component functions x¹, …, xⁿ of φ are the
    local coordinates.

    Required:
        domain   — the open subset U ⊂ M
        codomain — the open subset V ⊂ ℝⁿ
        inverse  — the smooth inverse φ⁻¹: V → U
        symbols  — ordered SymPy symbols (x¹, …, xⁿ) for this chart's coordinates
    """

    @property
    @abstractmethod
    def symbols(self) -> tuple[sympy.Symbol, ...]:
        """Ordered coordinate symbols (x¹, …, xⁿ) for this chart."""


class Atlas(IndexedFamily):
    """A family of charts constituting the smooth structure of a manifold.

    An atlas gives a manifold its smooth structure: transition maps φβ ∘ φα⁻¹
    between overlapping charts are required to be C∞ diffeomorphisms.

    An Atlas is an IndexedFamily of Charts: charts are retrievable by integer
    index, and the union of their domains covers all of M.

    Required:
        __getitem__  — retrieve the chart at index i
        __len__      — number of charts in this atlas
    """

    @abstractmethod
    def __getitem__(self, index: int) -> Chart:
        """Return the chart at *index*."""

    @abstractmethod
    def __len__(self) -> int:
        """Number of charts in this atlas."""


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


__all__ = ["Atlas", "Chart", "Diffeomorphism", "Manifold"]
