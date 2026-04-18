"""Tests for the manifold ABC hierarchy."""

from __future__ import annotations

import pytest

from cosmic_foundry.indexed_set import IndexedSet
from cosmic_foundry.pseudo_riemannian_manifold import PseudoRiemannianManifold
from cosmic_foundry.riemannian_manifold import RiemannianManifold
from cosmic_foundry.set import Set
from cosmic_foundry.smooth_manifold import SmoothManifold

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


# ---------------------------------------------------------------------------
# Concrete subclasses satisfy the hierarchy
# ---------------------------------------------------------------------------


class FlatR3(RiemannianManifold):
    @property
    def ndim(self) -> int:
        return 3

    @property
    def signature(self) -> tuple[int, int]:
        return (3, 0)


class MinkowskiR4(PseudoRiemannianManifold):
    @property
    def ndim(self) -> int:
        return 4

    @property
    def signature(self) -> tuple[int, int]:
        return (1, 3)


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
