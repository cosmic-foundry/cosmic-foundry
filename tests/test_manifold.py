"""Tests for the Manifold ABC."""

from __future__ import annotations

import pytest

from cosmic_foundry.continuous.manifold import Manifold
from cosmic_foundry.foundation.indexed_set import IndexedSet

# ---------------------------------------------------------------------------
# Assertion functions
# ---------------------------------------------------------------------------


def assert_manifold_is_abstract() -> None:
    with pytest.raises(TypeError):
        Manifold()  # type: ignore[abstract]


def assert_manifold_branch_disjoint_from_indexed_set_branch() -> None:
    assert not issubclass(Manifold, IndexedSet)
    assert not issubclass(IndexedSet, Manifold)


# ---------------------------------------------------------------------------
# Test wrappers
# ---------------------------------------------------------------------------


def test_manifold_branch_disjoint_from_indexed_set_branch() -> None:
    assert_manifold_branch_disjoint_from_indexed_set_branch()
