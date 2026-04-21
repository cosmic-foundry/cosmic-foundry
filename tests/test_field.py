"""Tests for the Field hierarchy."""

from __future__ import annotations

from typing import Any

from cosmic_foundry.continuous.field import (
    Field,
    SymmetricTensorField,
    TensorField,
)
from cosmic_foundry.continuous.manifold import Manifold


class _StubManifold(Manifold):
    @property
    def ndim(self) -> int:
        return 3

    @property
    def atlas(self) -> Any:
        raise NotImplementedError


_M = _StubManifold()


class _CustomTensor(TensorField):
    """Arbitrary (2, 1) tensor field."""

    @property
    def manifold(self) -> Manifold:
        return _M

    @property
    def tensor_type(self) -> tuple[int, int]:
        return (2, 1)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return None


class _SymmetricTensor(SymmetricTensorField):
    @property
    def manifold(self) -> Manifold:
        return _M

    def component(self, i: int, j: int) -> Field:
        val = 1.0 if i == j else 0.0

        class _C(Field):
            @property
            def manifold(self) -> Manifold:
                return _M

            def __call__(self, *a: Any, **kw: Any) -> float:
                return val

        return _C()  # type: ignore[return-value]

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return None


# ---------------------------------------------------------------------------
# manifold constraint
# ---------------------------------------------------------------------------


def test_tensor_field_manifold_is_manifold() -> None:
    assert isinstance(_CustomTensor().manifold, Manifold)
    assert isinstance(_SymmetricTensor().manifold, Manifold)


# ---------------------------------------------------------------------------
# tensor_type
# ---------------------------------------------------------------------------


def test_custom_tensor_type() -> None:
    assert _CustomTensor().tensor_type == (2, 1)


def test_tensor_type_is_nonneg_int_pair() -> None:
    for instance in [_CustomTensor(), _SymmetricTensor()]:
        p, q = instance.tensor_type
        assert isinstance(p, int) and isinstance(q, int)
        assert p >= 0 and q >= 0
