"""Schwarzschild spacetime: the unique static, spherically symmetric vacuum solution.

SchwarzschildSpacetime is a concrete PseudoRiemannianManifold. The metric is
expressed in Schwarzschild coordinates (t, r, θ, φ) with signature (-,+,+,+)
and geometric units G = c = 1. M is a free SymPy symbol; substituting a
numerical value yields a specific spacetime (Earth, Sun, black hole, …).
"""

from __future__ import annotations

import sympy

from cosmic_foundry.continuous.field import Field
from cosmic_foundry.continuous.manifold import Atlas, Chart
from cosmic_foundry.continuous.pseudo_riemannian_manifold import (
    MetricTensor,
    PseudoRiemannianManifold,
)

# ---------------------------------------------------------------------------
# Coordinate symbols — owned by the chart, shared by all component fields
# ---------------------------------------------------------------------------

t, r, theta, phi = sympy.symbols("t r theta phi", real=True)
M = sympy.Symbol("M", positive=True)  # geometric mass; G = c = 1

_SYMBOLS: tuple[sympy.Symbol, ...] = (t, r, theta, phi)

# ---------------------------------------------------------------------------
# Metric components  g_{μν}  in (−,+,+,+) convention
# ---------------------------------------------------------------------------

_rs = 2 * M  # Schwarzschild radius

_METRIC = sympy.Matrix(
    [
        [-(1 - _rs / r), 0, 0, 0],
        [0, 1 / (1 - _rs / r), 0, 0],
        [0, 0, r**2, 0],
        [0, 0, 0, r**2 * sympy.sin(theta) ** 2],
    ]
)


# ---------------------------------------------------------------------------
# Private scalar field: a single SymPy expression on the spacetime
# ---------------------------------------------------------------------------


class _ScalarField(Field):
    def __init__(self, expr: sympy.Expr, spacetime: SchwarzschildSpacetime) -> None:
        self._expr = expr
        self._spacetime = spacetime

    @property
    def expr(self) -> sympy.Expr:
        return self._expr

    @property
    def symbols(self) -> tuple[sympy.Symbol, ...]:
        return _SYMBOLS

    @property
    def manifold(self) -> SchwarzschildSpacetime:
        return self._spacetime


# ---------------------------------------------------------------------------
# Metric tensor
# ---------------------------------------------------------------------------


class SchwarzschildMetric(MetricTensor):
    def __init__(self, spacetime: SchwarzschildSpacetime) -> None:
        self._spacetime = spacetime

    @property
    def expr(self) -> sympy.Matrix:
        return _METRIC

    @property
    def symbols(self) -> tuple[sympy.Symbol, ...]:
        return _SYMBOLS

    @property
    def manifold(self) -> SchwarzschildSpacetime:
        return self._spacetime

    def as_matrix(self) -> sympy.Matrix:
        """Return the metric components as a SymPy Matrix."""
        return _METRIC

    def component(self, i: int, j: int) -> _ScalarField:
        return _ScalarField(_METRIC[i, j], self._spacetime)


# ---------------------------------------------------------------------------
# Chart: Schwarzschild coordinates (t, r, θ, φ)
# ---------------------------------------------------------------------------


class SchwarzschildChart(Chart):
    """Schwarzschild coordinate chart: φ: M → ℝ⁴ via (t, r, θ, φ).

    domain  — the Schwarzschild spacetime (r > r_s, exterior region)
    codomain — ℝ⁴; not yet represented as a concrete manifold object
    __call__ — point-to-coordinate map; not yet implemented
    """

    def __init__(self, spacetime: SchwarzschildSpacetime) -> None:
        self._spacetime = spacetime

    @property
    def symbols(self) -> tuple[sympy.Symbol, ...]:
        return _SYMBOLS

    @property
    def domain(self) -> SchwarzschildSpacetime:
        return self._spacetime

    @property
    def codomain(self) -> SchwarzschildSpacetime:
        raise NotImplementedError("EuclideanSpace codomain not yet represented")

    @property
    def inverse(self) -> SchwarzschildChart:
        raise NotImplementedError("inverse chart not yet implemented")

    def __call__(self, *args: object, **kwargs: object) -> object:
        raise NotImplementedError("point-to-coordinate map not yet implemented")


# ---------------------------------------------------------------------------
# Atlas: single-chart cover of the exterior region
# ---------------------------------------------------------------------------


class SchwarzschildAtlas(Atlas):
    def __init__(self, spacetime: SchwarzschildSpacetime) -> None:
        self._chart = SchwarzschildChart(spacetime)

    def __getitem__(self, index: int) -> SchwarzschildChart:
        if index != 0:
            raise IndexError(index)
        return self._chart

    def __len__(self) -> int:
        return 1


# ---------------------------------------------------------------------------
# Spacetime
# ---------------------------------------------------------------------------


class SchwarzschildSpacetime(PseudoRiemannianManifold):
    """The Schwarzschild vacuum solution.

    The Schwarzschild spacetime is the unique static, spherically symmetric
    solution to the vacuum Einstein field equations (Birkhoff's theorem). It
    describes the exterior geometry of any spherically symmetric mass
    distribution — black holes, neutron stars, and to excellent approximation,
    Earth and the Sun.

    Signature (1, 3): one timelike and three spacelike dimensions. The metric
    is expressed in geometric units (G = c = 1); M is a free SymPy symbol
    representing the central mass, so a single instance parametrises the full
    family of Schwarzschild spacetimes.
    """

    @property
    def signature(self) -> tuple[int, int]:
        return (1, 3)

    @property
    def metric(self) -> SchwarzschildMetric:
        return SchwarzschildMetric(self)

    @property
    def atlas(self) -> SchwarzschildAtlas:
        return SchwarzschildAtlas(self)


__all__ = [
    "M",
    "SchwarzschildMetric",
    "SchwarzschildSpacetime",
    "phi",
    "r",
    "t",
    "theta",
]
