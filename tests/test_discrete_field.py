"""Architectural constraints on the DiscreteField hierarchy."""

from __future__ import annotations

from typing import Any

from cosmic_foundry.continuous.differential_form import DifferentialForm
from cosmic_foundry.continuous.field import Field
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


class _ConcreteDifferentialForm(DifferentialForm):
    @property
    def degree(self) -> int:
        return 0

    @property
    def manifold(self) -> Manifold:
        return _M

    def __call__(self, *args: Any, **kwargs: Any) -> float:
        return 0.0


def test_discrete_field_is_not_continuous_field() -> None:
    assert not issubclass(DiscreteScalarField, Field)
    assert not issubclass(DiscreteVectorField, Field)


def test_discrete_scalar_approximates_narrows_to_differential_form() -> None:
    f = _DiscreteScalar(approximates=_ConcreteDifferentialForm())
    assert isinstance(f.approximates, DifferentialForm)
