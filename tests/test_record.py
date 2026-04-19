"""Tests for Array."""

from __future__ import annotations

import pytest

from cosmic_foundry.computation.array import Array


def test_array_getitem() -> None:
    a: Array[str] = Array(elements=("x", "y", "z"))
    assert a[0] == "x"
    assert a[1] == "y"
    assert a[2] == "z"


def test_array_as_dict() -> None:
    a: Array[int] = Array(elements=(10, 20))
    assert a.as_dict() == {"n": 2}


def test_array_rejects_empty_elements() -> None:
    with pytest.raises(ValueError, match="at least one"):
        Array(elements=())
