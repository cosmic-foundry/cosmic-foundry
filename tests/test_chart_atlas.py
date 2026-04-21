"""Tests for Chart, Atlas, IdentityChart, and SingleChartAtlas."""

from __future__ import annotations

import pytest

from cosmic_foundry.continuous.atlas import Atlas
from cosmic_foundry.continuous.chart import Chart
from cosmic_foundry.continuous.euclidean_space import EuclideanSpace
from cosmic_foundry.continuous.identity_chart import IdentityChart
from cosmic_foundry.continuous.minkowski_space import MinkowskiSpace
from cosmic_foundry.continuous.single_chart_atlas import SingleChartAtlas
from cosmic_foundry.continuous.smooth_manifold import SmoothManifold
from cosmic_foundry.foundation.function import Function
from cosmic_foundry.foundation.indexed_family import IndexedFamily

_E3 = EuclideanSpace(3)

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
    assert isinstance(IdentityChart(_E3), Function)


def assert_atlas_is_indexed_family() -> None:
    assert isinstance(SingleChartAtlas(IdentityChart(_E3)), IndexedFamily)


# ---------------------------------------------------------------------------
# Assertion functions — IdentityChart
# ---------------------------------------------------------------------------


def assert_identity_chart_domain() -> None:
    assert IdentityChart(_E3).domain is _E3


def assert_identity_chart_codomain_is_euclidean_space() -> None:
    assert isinstance(IdentityChart(_E3).codomain, EuclideanSpace)


def assert_identity_chart_codomain_dimension_matches_domain() -> None:
    assert IdentityChart(_E3).codomain.ndim == _E3.ndim


def assert_identity_chart_is_own_inverse() -> None:
    chart = IdentityChart(_E3)
    assert chart.inverse is chart


def assert_identity_chart_call_is_identity() -> None:
    chart = IdentityChart(_E3)
    obj = object()
    assert chart(obj) is obj


# ---------------------------------------------------------------------------
# Assertion functions — SingleChartAtlas
# ---------------------------------------------------------------------------


def assert_single_chart_atlas_len() -> None:
    assert len(SingleChartAtlas(IdentityChart(_E3))) == 1


def assert_single_chart_atlas_getitem_zero() -> None:
    chart = IdentityChart(_E3)
    assert SingleChartAtlas(chart)[0] is chart


def assert_single_chart_atlas_getitem_out_of_range() -> None:
    with pytest.raises(IndexError):
        SingleChartAtlas(IdentityChart(_E3))[1]


def assert_single_chart_atlas_manifold() -> None:
    assert SingleChartAtlas(IdentityChart(_E3)).manifold is _E3


# ---------------------------------------------------------------------------
# Assertion functions — SmoothManifold.atlas
# ---------------------------------------------------------------------------


def assert_smooth_manifold_atlas_is_abstract() -> None:
    with pytest.raises(TypeError):
        SmoothManifold()  # type: ignore[abstract]


def assert_euclidean_space_atlas_is_single_chart() -> None:
    atlas = _E3.atlas
    assert isinstance(atlas, SingleChartAtlas)
    assert len(atlas) == 1


def assert_euclidean_space_atlas_chart_domain_is_self() -> None:
    assert _E3.atlas[0].domain is _E3


def assert_euclidean_space_atlas_chart_codomain_ndim() -> None:
    assert _E3.atlas[0].codomain.ndim == 3


def assert_euclidean_space_atlas_is_atlas() -> None:
    assert isinstance(_E3.atlas, Atlas)


def assert_euclidean_space_atlas_manifold() -> None:
    assert _E3.atlas.manifold is _E3


def assert_minkowski_space_has_atlas() -> None:
    m = MinkowskiSpace()
    assert isinstance(m.atlas, Atlas)
    assert len(m.atlas) == 1
    assert m.atlas[0].codomain.ndim == 4


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


def test_identity_chart_domain() -> None:
    assert_identity_chart_domain()


def test_identity_chart_codomain_is_euclidean_space() -> None:
    assert_identity_chart_codomain_is_euclidean_space()


def test_identity_chart_codomain_dimension_matches_domain() -> None:
    assert_identity_chart_codomain_dimension_matches_domain()


def test_identity_chart_is_own_inverse() -> None:
    assert_identity_chart_is_own_inverse()


def test_identity_chart_call_is_identity() -> None:
    assert_identity_chart_call_is_identity()


def test_single_chart_atlas_len() -> None:
    assert_single_chart_atlas_len()


def test_single_chart_atlas_getitem_zero() -> None:
    assert_single_chart_atlas_getitem_zero()


def test_single_chart_atlas_getitem_out_of_range() -> None:
    assert_single_chart_atlas_getitem_out_of_range()


def test_single_chart_atlas_manifold() -> None:
    assert_single_chart_atlas_manifold()


def test_smooth_manifold_atlas_is_abstract() -> None:
    assert_smooth_manifold_atlas_is_abstract()


def test_euclidean_space_atlas_is_single_chart() -> None:
    assert_euclidean_space_atlas_is_single_chart()


def test_euclidean_space_atlas_chart_domain_is_self() -> None:
    assert_euclidean_space_atlas_chart_domain_is_self()


def test_euclidean_space_atlas_chart_codomain_ndim() -> None:
    assert_euclidean_space_atlas_chart_codomain_ndim()


def test_euclidean_space_atlas_is_atlas() -> None:
    assert_euclidean_space_atlas_is_atlas()


def test_euclidean_space_atlas_manifold() -> None:
    assert_euclidean_space_atlas_manifold()


def test_minkowski_space_has_atlas() -> None:
    assert_minkowski_space_has_atlas()
