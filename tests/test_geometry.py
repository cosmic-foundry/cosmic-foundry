"""Tests for the geometry module: EuclideanSpace and MinkowskiSpace."""

from __future__ import annotations

import pytest

from cosmic_foundry.theory.euclidean_space import EuclideanSpace
from cosmic_foundry.theory.flat_manifold import FlatManifold
from cosmic_foundry.theory.minkowski_space import MinkowskiSpace
from cosmic_foundry.theory.pseudo_riemannian_manifold import PseudoRiemannianManifold
from cosmic_foundry.theory.riemannian_manifold import RiemannianManifold
from cosmic_foundry.theory.smooth_manifold import SmoothManifold

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
    assert isinstance(e, EuclideanSpace)
    assert isinstance(e, RiemannianManifold)
    assert isinstance(e, FlatManifold)
    assert isinstance(e, PseudoRiemannianManifold)
    assert isinstance(e, SmoothManifold)


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
    assert isinstance(m, MinkowskiSpace)
    assert isinstance(m, FlatManifold)
    assert isinstance(m, PseudoRiemannianManifold)
    assert isinstance(m, SmoothManifold)


def test_minkowski_space_is_not_riemannian() -> None:
    assert not isinstance(MinkowskiSpace(), RiemannianManifold)
