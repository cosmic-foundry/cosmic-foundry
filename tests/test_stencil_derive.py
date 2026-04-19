"""Tests for parameterizable stencil derivation.

Tests verify that derive_laplacian_stencil(order, ndim) produces exact
rational weights for Laplacian stencils of any approximation order.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from cosmic_foundry.computation.stencil import derive_stencil


def test_order2_ndim1() -> None:
    """1D second-order Laplacian: weights [1, -2, 1] at offsets [-1, 0, 1]."""
    result = derive_stencil(deriv_order=2, approx_order=2, ndim=1)

    terms_dict = dict(result["terms"])
    assert terms_dict[(-1,)] == Fraction(1)
    assert terms_dict[(0,)] == Fraction(-2)
    assert terms_dict[(1,)] == Fraction(1)

    assert result["radii"] == (1,)
    assert result["approx_order"] == 2


def test_order2_ndim3() -> None:
    """3D second-order Laplacian: center -6, six face neighbors with weight 1."""
    result = derive_stencil(deriv_order=2, approx_order=2, ndim=3)

    terms_dict = dict(result["terms"])

    # Center: 3 × (-2) = -6
    assert terms_dict[(0, 0, 0)] == Fraction(-6)

    # Six face neighbors, each with weight 1
    face_neighbors = [
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    ]
    for offset in face_neighbors:
        assert terms_dict[offset] == Fraction(
            1
        ), f"offset {offset}: expected 1, got {terms_dict[offset]}"

    assert result["radii"] == (1, 1, 1)
    assert result["approx_order"] == 2


def test_order4_ndim3() -> None:
    """3D 4th-order Laplacian: verify weights for center, near-, and far-neighbors."""
    result = derive_stencil(deriv_order=2, approx_order=4, ndim=3)

    terms_dict = dict(result["terms"])

    # 1D weights for order-4 second derivative: [-1/12, 4/3, -5/2, 4/3, -1/12]
    # at points [-2, -1, 0, 1, 2].
    # For 3D, center = 3 × (-5/2) = -15/2, near-neighbors = 4/3 each,
    # far-neighbors = -1/12 each.

    # Center term
    assert terms_dict[(0, 0, 0)] == Fraction(-15, 2), f"center: {terms_dict[(0, 0, 0)]}"

    # Near-neighbors: ±1 in each axis
    near_neighbors = [
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    ]
    for offset in near_neighbors:
        assert terms_dict[offset] == Fraction(
            4, 3
        ), f"near-neighbor {offset}: {terms_dict[offset]}"

    # Far-neighbors: ±2 in each axis
    far_neighbors = [
        (2, 0, 0),
        (-2, 0, 0),
        (0, 2, 0),
        (0, -2, 0),
        (0, 0, 2),
        (0, 0, -2),
    ]
    for offset in far_neighbors:
        assert terms_dict[offset] == Fraction(
            -1, 12
        ), f"far-neighbor {offset}: {terms_dict[offset]}"

    assert result["radii"] == (2, 2, 2)
    assert result["approx_order"] == 4


def test_deriv1_order2_ndim1() -> None:
    """1D 1st derivative, 2nd-order: centered difference [-1/2, 0, 1/2]."""
    result = derive_stencil(deriv_order=1, approx_order=2, ndim=1)

    terms_dict = dict(result["terms"])
    assert terms_dict[(-1,)] == Fraction(-1, 2)
    assert terms_dict[(0,)] == Fraction(0)
    assert terms_dict[(1,)] == Fraction(1, 2)

    assert result["radii"] == (1,)
    assert result["deriv_order"] == 1
    assert result["approx_order"] == 2


def test_deriv1_order4_ndim1() -> None:
    """1D 1st derivative, 4th-order: [1/12, -2/3, 0, 2/3, -1/12]."""
    result = derive_stencil(deriv_order=1, approx_order=4, ndim=1)

    terms_dict = dict(result["terms"])
    assert terms_dict[(-2,)] == Fraction(1, 12)
    assert terms_dict[(-1,)] == Fraction(-2, 3)
    assert terms_dict[(0,)] == Fraction(0)
    assert terms_dict[(1,)] == Fraction(2, 3)
    assert terms_dict[(2,)] == Fraction(-1, 12)

    assert result["radii"] == (2,)
    assert result["deriv_order"] == 1
    assert result["approx_order"] == 4


def test_validation_too_few_points() -> None:
    """Verify that deriv_order > available points raises ValueError."""
    with pytest.raises(ValueError, match="need >"):
        derive_stencil(deriv_order=3, approx_order=2, ndim=1)


@pytest.mark.parametrize("deriv_order,approx_order", [(1, 2), (1, 4), (2, 2), (2, 4)])
def test_weights_sum_to_zero(deriv_order: int, approx_order: int) -> None:
    """Weights of any consistent finite-difference stencil must sum to zero.

    This is a necessary condition: applying the stencil to a constant field
    must return zero.
    """
    result = derive_stencil(deriv_order, approx_order, ndim=3)
    weight_sum = sum(w for _, w in result["terms"])
    assert weight_sum == 0, (
        f"deriv_order {deriv_order}, approx_order {approx_order}: "
        f"weights sum to {weight_sum}, expected 0"
    )
