"""Tests for MetricTensor."""

from __future__ import annotations

from typing import Any

import pytest

from cosmic_foundry.theory.euclidean_space import EuclideanSpace
from cosmic_foundry.theory.field import SymmetricTensorField
from cosmic_foundry.theory.metric_tensor import MetricTensor
from cosmic_foundry.theory.minkowski_space import MinkowskiSpace
from cosmic_foundry.theory.pseudo_riemannian_manifold import PseudoRiemannianManifold
from cosmic_foundry.theory.riemannian_manifold import RiemannianManifold


class _EuclideanMetric(MetricTensor):
    """Flat Riemannian metric on ℝ³ (identity matrix)."""

    @property
    def manifold(self) -> EuclideanSpace:
        return EuclideanSpace(3)

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        return None


class _MinkowskiMetric(MetricTensor):
    """Lorentzian metric on Minkowski space."""

    @property
    def manifold(self) -> MinkowskiSpace:
        return MinkowskiSpace()

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        return None


# ---------------------------------------------------------------------------
# Hierarchy
# ---------------------------------------------------------------------------


def test_metric_tensor_is_symmetric_tensor_field() -> None:
    assert issubclass(MetricTensor, SymmetricTensorField)


def test_metric_tensor_is_abstract() -> None:
    with pytest.raises(TypeError):
        MetricTensor()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# tensor_type inherited
# ---------------------------------------------------------------------------


def test_euclidean_metric_tensor_type() -> None:
    assert _EuclideanMetric().tensor_type == (0, 2)


def test_minkowski_metric_tensor_type() -> None:
    assert _MinkowskiMetric().tensor_type == (0, 2)


# ---------------------------------------------------------------------------
# manifold narrows to PseudoRiemannianManifold
# ---------------------------------------------------------------------------


def test_euclidean_metric_manifold_is_riemannian() -> None:
    m = _EuclideanMetric().manifold
    assert isinstance(m, RiemannianManifold)
    assert isinstance(m, PseudoRiemannianManifold)


def test_minkowski_metric_manifold_is_pseudo_riemannian() -> None:
    m = _MinkowskiMetric().manifold
    assert isinstance(m, PseudoRiemannianManifold)


def test_metric_manifold_carries_signature() -> None:
    assert _EuclideanMetric().manifold.signature == (3, 0)
    assert _MinkowskiMetric().manifold.signature == (1, 3)
