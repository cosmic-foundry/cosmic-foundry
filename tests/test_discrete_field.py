"""Tests for the DiscreteField hierarchy."""

from __future__ import annotations

from typing import Any

from cosmic_foundry.continuous.differential_form import DifferentialForm
from cosmic_foundry.continuous.field import Field, TensorField
from cosmic_foundry.continuous.manifold import Manifold
from cosmic_foundry.discrete.discrete_field import (
    DiscreteScalarField,
    DiscreteVectorField,
)
from cosmic_foundry.foundation.indexed_set import IndexedSet


class _StubManifold(Manifold):
    @property
    def ndim(self) -> int:
        return 3

    @property
    def atlas(self) -> Any:
        raise NotImplementedError


_M = _StubManifold()


class _Grid(IndexedSet):
    @property
    def ndim(self) -> int:
        return 3

    @property
    def shape(self) -> tuple[int, ...]:
        return (8, 8, 8)

    def intersect(self, other: IndexedSet) -> IndexedSet | None:
        return None


class _DiscreteScalar(DiscreteScalarField):
    def __init__(self, approximates: DifferentialForm | None = None) -> None:
        self._approximates = approximates

    @property
    def grid(self) -> _Grid:
        return _Grid()

    @property
    def approximates(self) -> DifferentialForm | None:
        return self._approximates

    def __call__(self, *args: Any, **kwargs: Any) -> float:
        return 0.0


class _DiscreteVector(DiscreteVectorField):
    def __init__(self, approximates: TensorField | None = None) -> None:
        self._approximates = approximates

    @property
    def grid(self) -> _Grid:
        return _Grid()

    @property
    def approximates(self) -> TensorField | None:
        return self._approximates

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return (0.0, 0.0, 0.0)


class _ConcreteDifferentialForm(DifferentialForm):
    @property
    def degree(self) -> int:
        return 0

    @property
    def manifold(self) -> Manifold:
        return _M

    def __call__(self, *args: Any, **kwargs: Any) -> float:
        return 0.0


# ---------------------------------------------------------------------------
# DiscreteField base
# ---------------------------------------------------------------------------


def test_discrete_field_is_not_continuous_field() -> None:
    f = _DiscreteScalar()
    assert not isinstance(f, Field)


def test_discrete_field_grid_returns_indexed_set() -> None:
    f = _DiscreteScalar()
    assert isinstance(f.grid, IndexedSet)


# ---------------------------------------------------------------------------
# approximates
# ---------------------------------------------------------------------------


def test_approximates_none_by_default() -> None:
    f = _DiscreteScalar()
    assert f.approximates is None


def test_approximates_set_to_continuous_field() -> None:
    continuous = _ConcreteDifferentialForm()
    f = _DiscreteScalar(approximates=continuous)
    assert f.approximates is continuous


def test_discrete_scalar_approximates_narrows_to_differential_form() -> None:
    continuous = _ConcreteDifferentialForm()
    f = _DiscreteScalar(approximates=continuous)
    assert isinstance(f.approximates, DifferentialForm)


def test_discrete_vector_approximates_is_none_by_default() -> None:
    f = _DiscreteVector()
    assert f.approximates is None
