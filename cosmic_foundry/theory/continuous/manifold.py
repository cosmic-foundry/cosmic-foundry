"""Manifold ABCs: diffeomorphism, chart, atlas, smooth structure, and Point.

Point is co-located here to resolve the mutual dependency: Point carries a
Chart, and Chart.__call__ maps a Point to coordinates.  With
``from __future__ import annotations`` both forward references are lazy and
the cycle never forms.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import sympy

from cosmic_foundry.theory.continuous.topological_manifold import TopologicalManifold
from cosmic_foundry.theory.foundation.homeomorphism import Homeomorphism
from cosmic_foundry.theory.foundation.indexed_family import IndexedFamily

D = TypeVar("D")
C = TypeVar("C")
M = TypeVar("M")


class Diffeomorphism(Homeomorphism[D, C]):
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


class Chart(Diffeomorphism[D, C]):
    """A local coordinate system on a smooth manifold: φ: U → V.

    A chart maps an open subset U of a manifold M diffeomorphically onto an
    open subset V of ℝⁿ. The component functions x¹, …, xⁿ of φ are the
    local coordinates.

    Required:
        domain   — the open subset U ⊂ M
        codomain — the open subset V ⊂ ℝⁿ
        inverse  — the smooth inverse φ⁻¹: V → U
        symbols  — ordered SymPy symbols (x¹, …, xⁿ) for this chart's coordinates
        __call__ — map a Point[D] on this manifold to its coordinate tuple
    """

    @property
    @abstractmethod
    def symbols(self) -> tuple[sympy.Symbol, ...]:
        """Ordered coordinate symbols (x¹, …, xⁿ) for this chart."""

    @abstractmethod
    def __call__(self, point: Point[D]) -> C:  # type: ignore[override]
        """Return the coordinates of point in this chart."""


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


@dataclass(frozen=True)
class Point(Generic[M]):
    """A point on manifold M expressed as coordinates in a specific chart.

    A manifold point is chart-independent mathematically, but to compute with
    it you must choose a chart and record which one.  Point carries both so
    that SymbolicFunction.__call__ can verify the chart matches the field's
    symbols and perform the correct substitution.

    Co-located with Chart in this module to resolve the mutual dependency:
    Point.chart: Chart[M, Any] and Chart.__call__(Point[D]) -> C reference
    each other.  Forward references under ``from __future__ import annotations``
    avoid any import cycle.

    frozen=True makes Point hashable so it can be used as a dict key or in
    sets — e.g. to cache field evaluations at grid nodes.  All fields must
    themselves be hashable for this to work in practice (coords should be
    SymPy expressions or Python scalars, not mutable containers).

    Required:
        manifold — the manifold this point belongs to
        chart    — the chart whose coordinate system the coords are expressed in
        coords   — coordinate values in the order defined by chart.symbols
    """

    manifold: M
    chart: Chart[M, Any]
    coords: tuple[Any, ...]


__all__ = ["Atlas", "Chart", "Diffeomorphism", "Manifold", "Point"]
