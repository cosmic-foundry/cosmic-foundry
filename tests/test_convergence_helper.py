"""Tests for the convergence-order measurement helper.

Reference problem: 1-D second-order centered finite difference of d²f/dx².

    f(x)  = sin(2πx)  on [0, 1]
    exact = -(2π)² sin(2πx)            (analytically derived)
    stencil: (f[i-1] - 2f[i] + f[i+1]) / h²   (O(h²) centered FD)

Validity conditions: smooth IC (no shocks, no limiter activation),
periodic-compatible boundary (f(0)=f(1)=0), conservative scheme.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tests.utils.convergence import assert_convergence_order, measure_convergence_order

# ---------------------------------------------------------------------------
# Reference error function: 2nd-order centered d²f/dx²
# ---------------------------------------------------------------------------

_TWO_PI = 2.0 * math.pi


def _second_order_l2_error(n: int) -> float:
    """L2 error of the 2nd-order centered stencil for d²f/dx² on n interior points.

    f(x) = sin(2πx); exact d²f/dx² = -(2π)²sin(2πx).
    Validity: smooth IC, no limiter, no outflow flux.
    """
    x = np.linspace(0.0, 1.0, n + 2)
    h = x[1] - x[0]
    f = np.sin(_TWO_PI * x)
    exact_interior = -(_TWO_PI**2) * f[1:-1]
    numerical = (f[:-2] - 2.0 * f[1:-1] + f[2:]) / h**2
    return float(np.sqrt(np.mean((numerical - exact_interior) ** 2)))


# First-order reference: forward-difference approximation of df/dx.
# exact df/dx = 2π cos(2πx); O(h¹) forward FD.
def _first_order_l2_error(n: int) -> float:
    x = np.linspace(0.0, 1.0, n + 2)
    h = x[1] - x[0]
    f = np.sin(_TWO_PI * x)
    exact_interior = _TWO_PI * np.cos(_TWO_PI * x[1:-1])
    numerical = (f[2:] - f[1:-1]) / h
    return float(np.sqrt(np.mean((numerical - exact_interior) ** 2)))


_RESOLUTIONS = [16, 32, 64, 128]

# ---------------------------------------------------------------------------
# Tests of the helper itself
# ---------------------------------------------------------------------------


def test_measure_convergence_order_second_order() -> None:
    order = measure_convergence_order(_second_order_l2_error, _RESOLUTIONS)
    assert abs(order - 2.0) <= 0.15, f"Measured order {order:.3f}, expected ~2.0"


def test_measure_convergence_order_first_order() -> None:
    order = measure_convergence_order(_first_order_l2_error, _RESOLUTIONS)
    assert abs(order - 1.0) <= 0.15, f"Measured order {order:.3f}, expected ~1.0"


def test_assert_convergence_order_passes() -> None:
    measured = assert_convergence_order(
        _second_order_l2_error, _RESOLUTIONS, expected=2.0
    )
    assert abs(measured - 2.0) <= 0.15


def test_assert_convergence_order_fails_for_wrong_expected() -> None:
    """Helper must reject a first-order scheme when second order is required."""
    with pytest.raises(AssertionError, match="Convergence order"):
        assert_convergence_order(
            _first_order_l2_error, _RESOLUTIONS, expected=2.0, atol=0.15
        )


def test_requires_minimum_three_resolutions() -> None:
    with pytest.raises(ValueError, match="3 resolutions"):
        measure_convergence_order(_second_order_l2_error, [32, 64])
