"""Tests for Chart and Atlas ABCs."""

from __future__ import annotations

from typing import Any

import pytest

from cosmic_foundry.continuous.atlas import Atlas
from cosmic_foundry.continuous.chart import Chart
from cosmic_foundry.continuous.manifold import Manifold
from cosmic_foundry.foundation.function import Function
from cosmic_foundry.foundation.indexed_family import IndexedFamily

# ---------------------------------------------------------------------------
# Minimal concrete stubs
# ---------------------------------------------------------------------------


class _StubManifold(Manifold):
    @property
    def ndim(self) -> int:
        return 3

    @property
    def atlas(self) -> Any:
        raise NotImplementedError


_m = _StubManifold()


class _StubChart(Chart[Any, Any]):
    @property
    def domain(self) -> Manifold:
        return _m

    @property
    def codomain(self) -> Manifold:
        return _m

    @property
    def inverse(self) -> Function[Any, Any]:
        return self

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return args[0] if args else None


class _StubAtlas(Atlas):
    @property
    def manifold(self) -> Manifold:
        return _m

    def __getitem__(self, index: int) -> Chart:
        if index != 0:
            raise IndexError(index)
        return _StubChart()

    def __len__(self) -> int:
        return 1


# ---------------------------------------------------------------------------
# Assertion functions — abstraction guards
# ---------------------------------------------------------------------------


def assert_chart_is_abstract() -> None:
    with pytest.raises(TypeError):
        Chart()  # type: ignore[abstract]


def assert_atlas_is_abstract() -> None:
    with pytest.raises(TypeError):
        Atlas()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# Assertion functions — inheritance
# ---------------------------------------------------------------------------


def assert_chart_is_function() -> None:
    assert isinstance(_StubChart(), Function)


def assert_atlas_is_indexed_family() -> None:
    assert isinstance(_StubAtlas(), IndexedFamily)


# ---------------------------------------------------------------------------
# Test wrappers
# ---------------------------------------------------------------------------


def test_chart_is_abstract() -> None:
    assert_chart_is_abstract()


def test_atlas_is_abstract() -> None:
    assert_atlas_is_abstract()


def test_chart_is_function() -> None:
    assert_chart_is_function()


def test_atlas_is_indexed_family() -> None:
    assert_atlas_is_indexed_family()
