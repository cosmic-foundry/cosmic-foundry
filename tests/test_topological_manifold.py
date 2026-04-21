"""Tests for the TopologicalManifold ABC."""

from __future__ import annotations

import pytest

from cosmic_foundry.continuous.topological_manifold import TopologicalManifold
from cosmic_foundry.foundation.set import Set
from cosmic_foundry.foundation.topological_space import TopologicalSpace


class _EuclideanSpace(TopologicalManifold):
    def __init__(self, n: int) -> None:
        self._n = n

    @property
    def ndim(self) -> int:
        return self._n


# ---------------------------------------------------------------------------
# Assertion functions
# ---------------------------------------------------------------------------


def assert_topological_manifold_is_abstract() -> None:
    with pytest.raises(TypeError):
        TopologicalManifold()  # type: ignore[abstract]


def assert_isinstance_chain() -> None:
    m = _EuclideanSpace(3)
    assert isinstance(m, TopologicalManifold)
    assert isinstance(m, TopologicalSpace)
    assert isinstance(m, Set)


def assert_ndim_is_int() -> None:
    for n in range(4):
        m = _EuclideanSpace(n)
        assert isinstance(m.ndim, int)
        assert m.ndim == n


# ---------------------------------------------------------------------------
# Test wrappers
# ---------------------------------------------------------------------------


def test_isinstance_chain() -> None:
    assert_isinstance_chain()


def test_ndim_is_int() -> None:
    assert_ndim_is_int()
