"""Tests for Chart, Atlas, IdentityChart, and SingleChartAtlas."""

from __future__ import annotations

import pytest

from cosmic_foundry.continuous.atlas import Atlas
from cosmic_foundry.continuous.chart import Chart
from cosmic_foundry.continuous.euclidean_space import EuclideanSpace
from cosmic_foundry.continuous.identity_chart import IdentityChart
from cosmic_foundry.continuous.single_chart_atlas import SingleChartAtlas
from cosmic_foundry.continuous.smooth_manifold import SmoothManifold
from cosmic_foundry.foundation.function import Function
from cosmic_foundry.foundation.indexed_family import IndexedFamily

_E3 = EuclideanSpace(3)
_E1 = EuclideanSpace(1)


# ---------------------------------------------------------------------------
# Abstraction guards
# ---------------------------------------------------------------------------


def test_chart_is_abstract() -> None:
    with pytest.raises(TypeError):
        Chart()  # type: ignore[abstract]


def test_atlas_is_abstract() -> None:
    with pytest.raises(TypeError):
        Atlas()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# Inheritance
# ---------------------------------------------------------------------------


def test_chart_is_function() -> None:
    chart = IdentityChart(_E3)
    assert isinstance(chart, Function)


def test_atlas_is_indexed_family() -> None:
    atlas = SingleChartAtlas(IdentityChart(_E3))
    assert isinstance(atlas, IndexedFamily)


# ---------------------------------------------------------------------------
# IdentityChart
# ---------------------------------------------------------------------------


def test_identity_chart_domain() -> None:
    chart = IdentityChart(_E3)
    assert chart.domain is _E3


def test_identity_chart_codomain_is_euclidean_space() -> None:
    chart = IdentityChart(_E3)
    assert isinstance(chart.codomain, EuclideanSpace)


def test_identity_chart_codomain_dimension_matches_domain() -> None:
    chart = IdentityChart(_E3)
    assert chart.codomain.ndim == _E3.ndim


def test_identity_chart_is_own_inverse() -> None:
    chart = IdentityChart(_E3)
    assert chart.inverse is chart


def test_identity_chart_call_is_identity() -> None:
    chart = IdentityChart(_E3)
    obj = object()
    assert chart(obj) is obj


# ---------------------------------------------------------------------------
# SingleChartAtlas
# ---------------------------------------------------------------------------


def test_single_chart_atlas_len() -> None:
    atlas = SingleChartAtlas(IdentityChart(_E3))
    assert len(atlas) == 1


def test_single_chart_atlas_getitem_zero() -> None:
    chart = IdentityChart(_E3)
    atlas = SingleChartAtlas(chart)
    assert atlas[0] is chart


def test_single_chart_atlas_getitem_out_of_range() -> None:
    atlas = SingleChartAtlas(IdentityChart(_E3))
    with pytest.raises(IndexError):
        atlas[1]


def test_single_chart_atlas_manifold() -> None:
    atlas = SingleChartAtlas(IdentityChart(_E3))
    assert atlas.manifold is _E3


# ---------------------------------------------------------------------------
# SmoothManifold.atlas
# ---------------------------------------------------------------------------


def test_smooth_manifold_atlas_is_abstract() -> None:
    with pytest.raises(TypeError):
        SmoothManifold()  # type: ignore[abstract]


def test_euclidean_space_atlas_is_single_chart() -> None:
    atlas = _E3.atlas
    assert isinstance(atlas, SingleChartAtlas)
    assert len(atlas) == 1


def test_euclidean_space_atlas_chart_domain_is_self() -> None:
    assert _E3.atlas[0].domain is _E3


def test_euclidean_space_atlas_chart_codomain_ndim() -> None:
    assert _E3.atlas[0].codomain.ndim == 3


def test_euclidean_space_atlas_is_atlas() -> None:
    assert isinstance(_E3.atlas, Atlas)


def test_euclidean_space_atlas_manifold() -> None:
    assert _E3.atlas.manifold is _E3


def test_minkowski_space_has_atlas() -> None:
    from cosmic_foundry.continuous.minkowski_space import MinkowskiSpace

    m = MinkowskiSpace()
    assert isinstance(m.atlas, Atlas)
    assert len(m.atlas) == 1
    assert m.atlas[0].codomain.ndim == 4
