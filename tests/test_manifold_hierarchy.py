"""Tests for the manifold ABC hierarchy and concrete manifolds."""

from __future__ import annotations

import pytest

from cosmic_foundry.continuous.euclidean_space import EuclideanSpace
from cosmic_foundry.continuous.flat_manifold import FlatManifold
from cosmic_foundry.continuous.identity_chart import IdentityChart
from cosmic_foundry.continuous.minkowski_space import MinkowskiSpace
from cosmic_foundry.continuous.pseudo_riemannian_manifold import (
    PseudoRiemannianManifold,
)
from cosmic_foundry.continuous.riemannian_manifold import RiemannianManifold
from cosmic_foundry.continuous.single_chart_atlas import SingleChartAtlas
from cosmic_foundry.continuous.smooth_manifold import SmoothManifold
from cosmic_foundry.foundation.indexed_set import IndexedSet
from cosmic_foundry.foundation.set import Set

# ---------------------------------------------------------------------------
# Instantiation guards
# ---------------------------------------------------------------------------


def test_smooth_manifold_is_abstract() -> None:
    with pytest.raises(TypeError):
        SmoothManifold()  # type: ignore[abstract]


def test_pseudo_riemannian_manifold_is_abstract() -> None:
    with pytest.raises(TypeError):
        PseudoRiemannianManifold()  # type: ignore[abstract]


def test_riemannian_manifold_is_abstract() -> None:
    with pytest.raises(TypeError):
        RiemannianManifold()  # type: ignore[abstract]


def test_flat_manifold_is_abstract() -> None:
    with pytest.raises(TypeError):
        FlatManifold()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# Concrete subclasses satisfy the hierarchy
# ---------------------------------------------------------------------------


class FlatLorentzian(FlatManifold):
    """Minimal flat Lorentzian manifold for hierarchy tests."""

    @property
    def signature(self) -> tuple[int, int]:
        return (1, 3)

    @property
    def atlas(self) -> SingleChartAtlas:
        return SingleChartAtlas(IdentityChart(self))


class FlatR3(RiemannianManifold):
    @property
    def ndim(self) -> int:
        return 3

    @property
    def signature(self) -> tuple[int, int]:
        return (3, 0)

    @property
    def atlas(self) -> SingleChartAtlas:
        return SingleChartAtlas(IdentityChart(self))


class MinkowskiR4(PseudoRiemannianManifold):
    @property
    def ndim(self) -> int:
        return 4

    @property
    def signature(self) -> tuple[int, int]:
        return (1, 3)

    @property
    def atlas(self) -> SingleChartAtlas:
        return SingleChartAtlas(IdentityChart(self))


def test_flat_r3_is_riemannian() -> None:
    m = FlatR3()
    assert m.ndim == 3
    assert m.signature == (3, 0)


def test_flat_r3_isinstance_chain() -> None:
    m = FlatR3()
    assert isinstance(m, RiemannianManifold)
    assert isinstance(m, PseudoRiemannianManifold)
    assert isinstance(m, SmoothManifold)
    assert isinstance(m, Set)


def test_flat_r3_is_not_indexed_set() -> None:
    assert not issubclass(RiemannianManifold, IndexedSet)


def test_minkowski_is_pseudo_riemannian_not_riemannian() -> None:
    m = MinkowskiR4()
    assert m.ndim == 4
    assert m.signature == (1, 3)
    assert isinstance(m, PseudoRiemannianManifold)
    assert not isinstance(m, RiemannianManifold)


def test_manifold_branch_disjoint_from_indexed_set_branch() -> None:
    assert not issubclass(SmoothManifold, IndexedSet)
    assert not issubclass(IndexedSet, SmoothManifold)


# ---------------------------------------------------------------------------
# FlatManifold hierarchy
# ---------------------------------------------------------------------------


def test_flat_lorentzian_isinstance_chain() -> None:
    m = FlatLorentzian()
    assert isinstance(m, FlatManifold)
    assert isinstance(m, PseudoRiemannianManifold)
    assert isinstance(m, SmoothManifold)
    assert isinstance(m, Set)


def test_flat_lorentzian_is_not_riemannian() -> None:
    assert not isinstance(FlatLorentzian(), RiemannianManifold)


def test_flat_lorentzian_ndim_derived_from_signature() -> None:
    assert FlatLorentzian().ndim == 4


def test_flat_manifold_is_pseudo_riemannian_not_riemannian() -> None:
    assert issubclass(FlatManifold, PseudoRiemannianManifold)
    assert not issubclass(FlatManifold, RiemannianManifold)


# ---------------------------------------------------------------------------
# EuclideanSpace
# ---------------------------------------------------------------------------


def test_euclidean_space_ndim() -> None:
    assert EuclideanSpace(3).ndim == 3


def test_euclidean_space_signature_derived() -> None:
    assert EuclideanSpace(3).signature == (3, 0)


def test_euclidean_space_ndim_matches_signature() -> None:
    e = EuclideanSpace(4)
    assert e.ndim == sum(e.signature)


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_euclidean_space_parametric(n: int) -> None:
    e = EuclideanSpace(n)
    assert e.ndim == n
    assert e.signature == (n, 0)


def test_euclidean_space_isinstance_chain() -> None:
    e = EuclideanSpace(3)
    assert isinstance(e, RiemannianManifold)
    assert isinstance(e, FlatManifold)
    assert isinstance(e, PseudoRiemannianManifold)
    assert isinstance(e, SmoothManifold)
    assert isinstance(e, Set)


def test_euclidean_space_invalid_dimension() -> None:
    with pytest.raises(ValueError):
        EuclideanSpace(0)


# ---------------------------------------------------------------------------
# MinkowskiSpace
# ---------------------------------------------------------------------------


def test_minkowski_space_signature() -> None:
    assert MinkowskiSpace().signature == (1, 3)


def test_minkowski_space_ndim_derived() -> None:
    assert MinkowskiSpace().ndim == 4


def test_minkowski_space_isinstance_chain() -> None:
    m = MinkowskiSpace()
    assert isinstance(m, FlatManifold)
    assert isinstance(m, PseudoRiemannianManifold)
    assert isinstance(m, SmoothManifold)
    assert isinstance(m, Set)


def test_minkowski_space_is_not_riemannian() -> None:
    assert not isinstance(MinkowskiSpace(), RiemannianManifold)
