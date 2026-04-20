"""Tests for the Field hierarchy."""

from __future__ import annotations

from typing import Any

import pytest

from cosmic_foundry.theory.euclidean_space import EuclideanSpace
from cosmic_foundry.theory.field import (
    CovectorField,
    Field,
    ScalarField,
    SymmetricTensorField,
    TensorField,
    VectorField,
)
from cosmic_foundry.theory.function import Function
from cosmic_foundry.theory.manifold import Manifold
from cosmic_foundry.theory.smooth_manifold import SmoothManifold

# ---------------------------------------------------------------------------
# Minimal concrete stubs
# ---------------------------------------------------------------------------

_M = EuclideanSpace(3)


class _Scalar(ScalarField):
    @property
    def manifold(self) -> EuclideanSpace:
        return _M

    def __call__(self, *args: Any, **kwargs: Any) -> float:
        return 0.0


class _Vector(VectorField):
    @property
    def manifold(self) -> EuclideanSpace:
        return _M

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return None


class _Covector(CovectorField):
    @property
    def manifold(self) -> EuclideanSpace:
        return _M

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return None


class _CustomTensor(TensorField):
    """Arbitrary (2, 1) tensor field."""

    @property
    def manifold(self) -> EuclideanSpace:
        return _M

    @property
    def tensor_type(self) -> tuple[int, int]:
        return (2, 1)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return None


class _SymmetricTensor(SymmetricTensorField):
    @property
    def manifold(self) -> EuclideanSpace:
        return _M

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return None


class _TopologicalField(Field):
    """A field on a bare topological manifold — no tensor structure."""

    class _TopoManifold(Manifold):
        @property
        def ndim(self) -> int:
            return 2

    @property
    def manifold(self) -> _TopoManifold:
        return self._TopoManifold()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return None


# ---------------------------------------------------------------------------
# Hierarchy
# ---------------------------------------------------------------------------


def test_field_is_function() -> None:
    assert issubclass(Field, Function)


def test_tensor_field_is_field() -> None:
    assert issubclass(TensorField, Field)


def test_scalar_field_is_tensor_field() -> None:
    assert issubclass(ScalarField, TensorField)


def test_vector_field_is_tensor_field() -> None:
    assert issubclass(VectorField, TensorField)


def test_covector_field_is_tensor_field() -> None:
    assert issubclass(CovectorField, TensorField)


def test_symmetric_tensor_field_is_tensor_field() -> None:
    assert issubclass(SymmetricTensorField, TensorField)


def test_field_is_abstract() -> None:
    with pytest.raises(TypeError):
        Field()  # type: ignore[abstract]


def test_tensor_field_is_abstract() -> None:
    with pytest.raises(TypeError):
        TensorField()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# manifold constraint
# ---------------------------------------------------------------------------


def test_field_manifold_can_be_topological() -> None:
    f = _TopologicalField()
    assert isinstance(f.manifold, Manifold)
    assert not isinstance(f.manifold, SmoothManifold)


def test_tensor_field_manifold_is_smooth() -> None:
    assert isinstance(_Scalar().manifold, SmoothManifold)
    assert isinstance(_Vector().manifold, SmoothManifold)
    assert isinstance(_Covector().manifold, SmoothManifold)


# ---------------------------------------------------------------------------
# tensor_type on named subclasses
# ---------------------------------------------------------------------------


def test_scalar_field_tensor_type() -> None:
    assert _Scalar().tensor_type == (0, 0)


def test_vector_field_tensor_type() -> None:
    assert _Vector().tensor_type == (1, 0)


def test_covector_field_tensor_type() -> None:
    assert _Covector().tensor_type == (0, 1)


def test_custom_tensor_type() -> None:
    assert _CustomTensor().tensor_type == (2, 1)


def test_symmetric_tensor_field_tensor_type() -> None:
    assert _SymmetricTensor().tensor_type == (0, 2)


def test_tensor_type_is_nonneg_int_pair() -> None:
    for instance in [
        _Scalar(),
        _Vector(),
        _Covector(),
        _CustomTensor(),
        _SymmetricTensor(),
    ]:
        p, q = instance.tensor_type
        assert isinstance(p, int) and isinstance(q, int)
        assert p >= 0 and q >= 0


# ---------------------------------------------------------------------------
# Callability
# ---------------------------------------------------------------------------


def test_scalar_field_callable() -> None:
    assert _Scalar()() == 0.0
