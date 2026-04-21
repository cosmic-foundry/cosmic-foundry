"""Tests for MetricTensor, EuclideanMetric, and MinkowskiMetric."""

from __future__ import annotations

import pytest
import sympy

from cosmic_foundry.continuous.euclidean_metric import EuclideanMetric
from cosmic_foundry.continuous.euclidean_space import EuclideanSpace
from cosmic_foundry.continuous.field import SymmetricTensorField
from cosmic_foundry.continuous.metric_tensor import MetricTensor
from cosmic_foundry.continuous.minkowski_metric import MinkowskiMetric
from cosmic_foundry.continuous.minkowski_space import MinkowskiSpace
from cosmic_foundry.continuous.pseudo_riemannian_manifold import (
    PseudoRiemannianManifold,
)
from cosmic_foundry.continuous.riemannian_manifold import RiemannianManifold

# ---------------------------------------------------------------------------
# Parameter sets
# ---------------------------------------------------------------------------

EUCLIDEAN_METRIC_DIMS = [1, 2, 3, 4]

# ---------------------------------------------------------------------------
# Assertion functions — MetricTensor hierarchy
# ---------------------------------------------------------------------------


def assert_metric_tensor_is_symmetric_tensor_field() -> None:
    assert issubclass(MetricTensor, SymmetricTensorField)


def assert_metric_tensor_is_abstract() -> None:
    with pytest.raises(TypeError):
        MetricTensor()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# Assertion functions — EuclideanMetric
# ---------------------------------------------------------------------------


def assert_euclidean_metric_is_metric_tensor() -> None:
    assert isinstance(EuclideanMetric(EuclideanSpace(3)), MetricTensor)


def assert_euclidean_metric_tensor_type() -> None:
    assert EuclideanMetric(EuclideanSpace(3)).tensor_type == (0, 2)


def assert_euclidean_metric_manifold_is_riemannian() -> None:
    m = EuclideanMetric(EuclideanSpace(3)).manifold
    assert isinstance(m, RiemannianManifold)
    assert isinstance(m, PseudoRiemannianManifold)


def assert_euclidean_metric_parametric(n: int) -> None:
    g = EuclideanMetric(EuclideanSpace(n))
    assert g() == sympy.eye(n)


def assert_euclidean_metric_manifold_signature() -> None:
    assert EuclideanMetric(EuclideanSpace(3)).manifold.signature == (3, 0)


# ---------------------------------------------------------------------------
# Assertion functions — MinkowskiMetric
# ---------------------------------------------------------------------------


def assert_minkowski_metric_is_metric_tensor() -> None:
    assert isinstance(MinkowskiMetric(MinkowskiSpace()), MetricTensor)


def assert_minkowski_metric_tensor_type() -> None:
    assert MinkowskiMetric(MinkowskiSpace()).tensor_type == (0, 2)


def assert_minkowski_metric_manifold_is_pseudo_riemannian() -> None:
    m = MinkowskiMetric(MinkowskiSpace()).manifold
    assert isinstance(m, PseudoRiemannianManifold)


def assert_minkowski_metric_components() -> None:
    assert MinkowskiMetric(MinkowskiSpace())() == sympy.diag(1, -1, -1, -1)


def assert_minkowski_metric_manifold_signature() -> None:
    assert MinkowskiMetric(MinkowskiSpace()).manifold.signature == (1, 3)


# ---------------------------------------------------------------------------
# Assertion functions — EuclideanSpace.metric and MinkowskiSpace.metric
# ---------------------------------------------------------------------------


def assert_euclidean_space_metric_is_euclidean_metric() -> None:
    assert isinstance(EuclideanSpace(3).metric, EuclideanMetric)


def assert_euclidean_space_metric_components() -> None:
    assert EuclideanSpace(3).metric() == sympy.eye(3)


def assert_minkowski_space_metric_is_minkowski_metric() -> None:
    assert isinstance(MinkowskiSpace().metric, MinkowskiMetric)


def assert_minkowski_space_metric_components() -> None:
    assert MinkowskiSpace().metric() == sympy.diag(1, -1, -1, -1)


# ---------------------------------------------------------------------------
# Test wrappers
# ---------------------------------------------------------------------------


def test_metric_tensor_is_symmetric_tensor_field() -> None:
    assert_metric_tensor_is_symmetric_tensor_field()


def test_metric_tensor_is_abstract() -> None:
    assert_metric_tensor_is_abstract()


def test_euclidean_metric_is_metric_tensor() -> None:
    assert_euclidean_metric_is_metric_tensor()


def test_euclidean_metric_tensor_type() -> None:
    assert_euclidean_metric_tensor_type()


def test_euclidean_metric_manifold_is_riemannian() -> None:
    assert_euclidean_metric_manifold_is_riemannian()


@pytest.mark.parametrize("n", EUCLIDEAN_METRIC_DIMS)
def test_euclidean_metric_parametric(n: int) -> None:
    assert_euclidean_metric_parametric(n)


def test_euclidean_metric_manifold_signature() -> None:
    assert_euclidean_metric_manifold_signature()


def test_minkowski_metric_is_metric_tensor() -> None:
    assert_minkowski_metric_is_metric_tensor()


def test_minkowski_metric_tensor_type() -> None:
    assert_minkowski_metric_tensor_type()


def test_minkowski_metric_manifold_is_pseudo_riemannian() -> None:
    assert_minkowski_metric_manifold_is_pseudo_riemannian()


def test_minkowski_metric_components() -> None:
    assert_minkowski_metric_components()


def test_minkowski_metric_manifold_signature() -> None:
    assert_minkowski_metric_manifold_signature()


def test_euclidean_space_metric_is_euclidean_metric() -> None:
    assert_euclidean_space_metric_is_euclidean_metric()


def test_euclidean_space_metric_components() -> None:
    assert_euclidean_space_metric_components()


def test_minkowski_space_metric_is_minkowski_metric() -> None:
    assert_minkowski_space_metric_is_minkowski_metric()


def test_minkowski_space_metric_components() -> None:
    assert_minkowski_space_metric_components()
