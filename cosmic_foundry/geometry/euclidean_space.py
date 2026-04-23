"""EuclideanSpace and CartesianChart: flat Euclidean geometry.

EuclideanSpace  — concrete RiemannianManifold representing ℝⁿ with flat metric
CartesianChart  — the identity chart on EuclideanSpace

These two classes are co-located because they are mutually dependent:
EuclideanSpace carries an atlas of CartesianCharts, and CartesianChart
carries its EuclideanSpace as domain.  The dependency is resolved by
lazy atlas construction.
"""

from __future__ import annotations

from typing import Any

import sympy

from cosmic_foundry.theory.continuous.field import Field
from cosmic_foundry.theory.continuous.manifold import Atlas, Chart, Manifold
from cosmic_foundry.theory.continuous.pseudo_riemannian_manifold import (
    MetricTensor,
    RiemannianManifold,
)

_NAMED_SYMBOL_NAMES = ("x", "y", "z")


class _ConstantField(Field):
    """A coordinate-independent scalar field with a fixed SymPy value."""

    def __init__(self, manifold: Manifold, value: sympy.Expr) -> None:
        self._manifold = manifold
        self._value = value

    @property
    def manifold(self) -> Manifold:
        return self._manifold

    @property
    def expr(self) -> sympy.Expr:
        return self._value

    @property
    def symbols(self) -> tuple[sympy.Symbol, ...]:
        return ()


class _CartesianMetric(MetricTensor):
    """The flat Euclidean metric expressed in Cartesian coordinates: g_ij = δ_ij.

    This is the coordinate representation of the flat metric in the
    Cartesian chart (the identity chart on EuclideanSpace).  In Cartesian
    coordinates all components are constants (no coordinate dependence), so
    g_ij = δ_ij exactly.

    Note: in a non-Cartesian chart on the same EuclideanSpace (e.g. polar or
    cylindrical), the metric components are not δ_ij — for example in 2-D
    polar coordinates g_θθ = r².  A different MetricTensor implementation
    would be required for those charts.
    """

    def __init__(self, space: EuclideanSpace) -> None:
        self._space = space

    @property
    def manifold(self) -> EuclideanSpace:
        return self._space

    @property
    def expr(self) -> sympy.Expr:
        return sympy.Integer(1)

    @property
    def symbols(self) -> tuple[sympy.Symbol, ...]:
        return self._space.symbols

    def component(self, i: int, j: int) -> Field:
        """Return the (i, j) Cartesian metric component: 1 if i == j, else 0."""
        value = sympy.Integer(1) if i == j else sympy.Integer(0)
        return _ConstantField(self._space, value)


class _CartesianAtlas(Atlas):
    """Single-chart atlas containing the identity CartesianChart."""

    def __init__(self, chart: CartesianChart) -> None:
        self._chart = chart

    def __getitem__(self, index: int) -> CartesianChart:
        if index != 0:
            raise IndexError(index)
        return self._chart

    def __len__(self) -> int:
        return 1


class CartesianChart(Chart):
    """The identity coordinate chart on EuclideanSpace: φ = id.

    A CartesianChart is the canonical chart on flat Euclidean space — the
    identity map φ: ℝⁿ → ℝⁿ.  It is its own inverse, and its symbols are
    the coordinate names of the EuclideanSpace it belongs to.

    Required:
        space — the EuclideanSpace this chart covers
    """

    def __init__(self, space: EuclideanSpace) -> None:
        self._space = space

    @property
    def domain(self) -> EuclideanSpace:
        """The EuclideanSpace this chart maps from."""
        return self._space

    @property
    def codomain(self) -> EuclideanSpace:
        """The EuclideanSpace this chart maps to (same space — identity map)."""
        return self._space

    @property
    def inverse(self) -> CartesianChart:
        """The inverse chart; the identity is its own inverse."""
        return self

    @property
    def symbols(self) -> tuple[sympy.Symbol, ...]:
        """Ordered coordinate symbols (x, y, …) for this chart."""
        return self._space.symbols

    def __call__(self, *args: Any, **kwargs: Any) -> tuple[Any, ...]:
        """Apply the identity map: return coordinates unchanged."""
        return args


class EuclideanSpace(RiemannianManifold):
    """Flat Euclidean space ℝⁿ with the standard Cartesian metric g = I.

    EuclideanSpace is the concrete domain for Cartesian geometry.  Its
    metric is the identity (g_ij = δ_ij), its atlas consists of a single
    CartesianChart (the identity map), and its coordinate symbols are
    derived from ndim.

    Free:
        ndim         — number of spatial dimensions
        symbol_names — optional coordinate names; defaults to (x, y, z) for
                       ndim ≤ 3, then (x0, x1, x2, …) for higher dimensions

    Derived:
        signature  — always (ndim, 0) for a Riemannian manifold
        symbols    — SymPy symbols for the coordinate names
        metric     — flat metric in Cartesian coordinates: g_ij = δ_ij
        atlas      — single-chart atlas containing the identity CartesianChart
    """

    def __init__(
        self,
        ndim: int,
        symbol_names: tuple[str, ...] | None = None,
    ) -> None:
        if symbol_names is None:
            if ndim <= len(_NAMED_SYMBOL_NAMES):
                symbol_names = _NAMED_SYMBOL_NAMES[:ndim]
            else:
                symbol_names = tuple(f"x{i}" for i in range(ndim))
        if len(symbol_names) != ndim:
            msg = f"len(symbol_names)={len(symbol_names)} must equal ndim={ndim}"
            raise ValueError(msg)
        self._ndim = ndim
        self._symbols = tuple(sympy.Symbol(n) for n in symbol_names)
        self._atlas: _CartesianAtlas | None = None
        self._metric: _CartesianMetric | None = None

    @property
    def ndim(self) -> int:
        return self._ndim

    @property
    def symbols(self) -> tuple[sympy.Symbol, ...]:
        """Coordinate symbols (x, y, …) for this space."""
        return self._symbols

    @property
    def atlas(self) -> _CartesianAtlas:
        """Single-chart atlas; constructed lazily to avoid circular init."""
        if self._atlas is None:
            self._atlas = _CartesianAtlas(CartesianChart(self))
        return self._atlas

    @property
    def metric(self) -> _CartesianMetric:
        """Flat metric in Cartesian coordinates: g_ij = δ_ij; constructed lazily."""
        if self._metric is None:
            self._metric = _CartesianMetric(self)
        return self._metric


__all__ = ["CartesianChart", "EuclideanSpace"]
