"""Tests for MetricTensor."""

from __future__ import annotations

import pytest

from cosmic_foundry.continuous.field import SymmetricTensorField
from cosmic_foundry.continuous.metric_tensor import MetricTensor


def test_metric_tensor_is_symmetric_tensor_field() -> None:
    assert issubclass(MetricTensor, SymmetricTensorField)


def test_metric_tensor_is_abstract() -> None:
    with pytest.raises(TypeError):
        MetricTensor()  # type: ignore[abstract]
