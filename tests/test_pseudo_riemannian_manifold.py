"""Tests for the PseudoRiemannianManifold ABC."""

from __future__ import annotations

from typing import Any

import pytest
import sympy

from cosmic_foundry.continuous.atlas import Atlas
from cosmic_foundry.continuous.chart import Chart
from cosmic_foundry.continuous.manifold import Manifold
from cosmic_foundry.continuous.metric_tensor import MetricTensor
from cosmic_foundry.continuous.pseudo_riemannian_manifold import (
    PseudoRiemannianManifold,
)
from cosmic_foundry.foundation.set import Set

# ---------------------------------------------------------------------------
# Minimal private helpers
# ---------------------------------------------------------------------------


class _StubAtlas(Atlas):
    def __init__(self, m: PseudoRiemannianManifold) -> None:
        self._m = m

    @property
    def manifold(self) -> PseudoRiemannianManifold:
        return self._m

    def __getitem__(self, index: int) -> Chart:
        raise IndexError(index)

    def __len__(self) -> int:
        return 0


class _StubMetric(MetricTensor):
    def __init__(self, m: PseudoRiemannianManifold) -> None:
        self._m = m

    @property
    def manifold(self) -> PseudoRiemannianManifold:
        return self._m

    def __call__(self, *args: Any, **kwargs: Any) -> sympy.Matrix:
        return sympy.eye(self._m.ndim)


# ---------------------------------------------------------------------------
# Concrete implementations
# ---------------------------------------------------------------------------


class FlatR3(PseudoRiemannianManifold):
    @property
    def signature(self) -> tuple[int, int]:
        return (3, 0)

    @property
    def metric(self) -> _StubMetric:
        return _StubMetric(self)

    @property
    def atlas(self) -> _StubAtlas:
        return _StubAtlas(self)


class MinkowskiR4(PseudoRiemannianManifold):
    @property
    def signature(self) -> tuple[int, int]:
        return (1, 3)

    @property
    def metric(self) -> _StubMetric:
        return _StubMetric(self)

    @property
    def atlas(self) -> _StubAtlas:
        return _StubAtlas(self)


# ---------------------------------------------------------------------------
# Assertion functions
# ---------------------------------------------------------------------------


def assert_pseudo_riemannian_manifold_is_abstract() -> None:
    with pytest.raises(TypeError):
        PseudoRiemannianManifold()  # type: ignore[abstract]


def assert_flat_r3_isinstance_chain() -> None:
    m = FlatR3()
    assert isinstance(m, PseudoRiemannianManifold)
    assert isinstance(m, Manifold)
    assert isinstance(m, Set)


def assert_flat_r3_ndim_derived_from_signature() -> None:
    m = FlatR3()
    assert m.signature == (3, 0)
    assert m.ndim == 3


def assert_minkowski_r4_isinstance_chain() -> None:
    m = MinkowskiR4()
    assert isinstance(m, PseudoRiemannianManifold)
    assert isinstance(m, Manifold)
    assert isinstance(m, Set)


def assert_minkowski_r4_ndim_derived_from_signature() -> None:
    m = MinkowskiR4()
    assert m.signature == (1, 3)
    assert m.ndim == 4


# ---------------------------------------------------------------------------
# Test wrappers
# ---------------------------------------------------------------------------


def test_pseudo_riemannian_manifold_is_abstract() -> None:
    assert_pseudo_riemannian_manifold_is_abstract()


def test_flat_r3_isinstance_chain() -> None:
    assert_flat_r3_isinstance_chain()


def test_flat_r3_ndim_derived_from_signature() -> None:
    assert_flat_r3_ndim_derived_from_signature()


def test_minkowski_r4_isinstance_chain() -> None:
    assert_minkowski_r4_isinstance_chain()


def test_minkowski_r4_ndim_derived_from_signature() -> None:
    assert_minkowski_r4_ndim_derived_from_signature()
