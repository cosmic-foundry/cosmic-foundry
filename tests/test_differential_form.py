"""Tests for DifferentialForm."""

from __future__ import annotations

from typing import Any

from cosmic_foundry.continuous.differential_form import DifferentialForm
from cosmic_foundry.continuous.manifold import Manifold


class _StubManifold(Manifold):
    @property
    def ndim(self) -> int:
        return 3

    @property
    def atlas(self) -> Any:
        raise NotImplementedError


_M = _StubManifold()


class _Form(DifferentialForm):
    def __init__(self, degree: int) -> None:
        self._degree = degree

    @property
    def degree(self) -> int:
        return self._degree

    @property
    def manifold(self) -> Manifold:
        return _M

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        return None


def test_tensor_type_tracks_degree() -> None:
    for k in range(4):
        form = _Form(k)
        assert form.tensor_type == (0, k)


def test_manifold_is_manifold() -> None:
    assert isinstance(_Form(2).manifold, Manifold)
