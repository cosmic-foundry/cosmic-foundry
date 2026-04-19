"""Tests for Array and Placement."""

from __future__ import annotations

import pytest

from cosmic_foundry.computation.array import Array, Placement

# ---------------------------------------------------------------------------
# Placement.component_ids
# ---------------------------------------------------------------------------


def test_placement_component_ids() -> None:
    p = Placement({0: 0, 1: 1, 2: 0})
    assert p.component_ids == frozenset({0, 1, 2})


# ---------------------------------------------------------------------------
# Array construction and lookup
# ---------------------------------------------------------------------------


def test_array_getitem() -> None:
    a: Array[str] = Array(
        elements=("x", "y", "z"),
        placement=Placement({0: 0, 1: 0, 2: 1}),
    )
    assert a[0] == "x"
    assert a[1] == "y"
    assert a[2] == "z"


def test_array_local_returns_elements_for_rank_in_index_order() -> None:
    a: Array[str] = Array(
        elements=("a", "b", "c", "d"),
        placement=Placement({0: 0, 1: 1, 2: 0, 3: 1}),
    )
    assert a.local(0) == ("a", "c")
    assert a.local(1) == ("b", "d")
    assert a.local(99) == ()


def test_array_as_dict() -> None:
    a: Array[int] = Array(
        elements=(10, 20),
        placement=Placement({0: 0, 1: 0}),
    )
    d = a.as_dict()
    assert d["n"] == 2
    assert "placement" in d


# ---------------------------------------------------------------------------
# Array validation
# ---------------------------------------------------------------------------


def test_array_rejects_placement_missing_index() -> None:
    with pytest.raises(ValueError, match="do not match"):
        Array(
            elements=("x", "y"),
            placement=Placement({0: 0}),
        )


def test_array_rejects_placement_with_extra_index() -> None:
    with pytest.raises(ValueError, match="do not match"):
        Array(
            elements=("x",),
            placement=Placement({0: 0, 1: 0}),
        )


def test_array_rejects_empty_elements() -> None:
    with pytest.raises(ValueError):
        Array(elements=(), placement=Placement({0: 0}))
