"""Architectural constraints on the Manifold branch."""

from __future__ import annotations

from cosmic_foundry.continuous.manifold import Manifold
from cosmic_foundry.foundation.indexed_set import IndexedSet


def test_manifold_branch_disjoint_from_indexed_set_branch() -> None:
    assert not issubclass(Manifold, IndexedSet)
    assert not issubclass(IndexedSet, Manifold)
