"""SymPy-based stencil coefficient verification.

Tests here verify that implemented kernel weights exactly match the
coefficients derived from first principles via Taylor-expansion (SymPy).
This is complementary to the analytical-solution tests in test_kernels.py:
those confirm the kernel produces correct *output* on a specific function;
these confirm the kernel encodes correct *weights*, catching coefficient bugs
that happen to cancel on simple smooth inputs.

Example of a bug the analytical test misses but probing catches
---------------------------------------------------------------
If the center weight were -5 instead of -6, the kernel applied to
phi = x²+y²+z² would yield 5 at interior points rather than 6 — which IS
caught by the existing test.  But if the center were -6 and one neighbor
coefficient were 0 instead of 1, the analytical test (which uses a
quadratic) would also fail.  The probing test would locate the specific
broken offset, which significantly speeds up diagnosis.

The real value of probing emerges with higher-order stencils (4th, 6th
order) where rational coefficients like -1/12 and 4/3 must be exact; an
error of 1 ULP would pass convergence tests at modest resolutions but
produce wrong answers at high resolution.
"""

from __future__ import annotations

import pytest
from sympy import Rational

from tests.utils.stencils import fd_coefficients, probe_operator_weights

# ---------------------------------------------------------------------------
# Tests of fd_coefficients (SymPy oracle)
# ---------------------------------------------------------------------------


def test_fd_coefficients_2nd_order_2nd_deriv() -> None:
    """2nd-order centered stencil for d²f/dx²: coefficients are [1, -2, 1]."""
    coeffs = fd_coefficients(2, [-1, 0, 1])
    assert coeffs == [Rational(1), Rational(-2), Rational(1)]


def test_fd_coefficients_4th_order_2nd_deriv() -> None:
    """4th-order centered stencil for d²f/dx²: [-1/12, 4/3, -5/2, 4/3, -1/12]."""
    coeffs = fd_coefficients(2, [-2, -1, 0, 1, 2])
    assert coeffs == [
        Rational(-1, 12),
        Rational(4, 3),
        Rational(-5, 2),
        Rational(4, 3),
        Rational(-1, 12),
    ]


def test_fd_coefficients_2nd_order_1st_deriv() -> None:
    """2nd-order centered stencil for df/dx: [-1/2, 0, 1/2]."""
    coeffs = fd_coefficients(1, [-1, 0, 1])
    assert coeffs == [Rational(-1, 2), Rational(0), Rational(1, 2)]


def test_fd_coefficients_requires_enough_offsets() -> None:
    with pytest.raises(ValueError, match="deriv_order"):
        fd_coefficients(2, [-1, 0])  # need ≥ 3 offsets for 2nd deriv


def test_fd_coefficients_rejects_zeroth_derivative() -> None:
    with pytest.raises(ValueError, match="deriv_order"):
        fd_coefficients(0, [-1, 0, 1])


# ---------------------------------------------------------------------------
# Stencil verification: 7-point Laplacian
# ---------------------------------------------------------------------------
# The seven_point_laplacian kernel (defined in test_kernels.py and used in
# production) returns the UN-divided stencil (no h² denominator).  At unit
# grid spacing h=1, divided and un-divided are numerically identical, so
# the SymPy rational coefficients apply directly.
#
# Validity: linear kernel, unit spacing, interior probe point with halo ≥ 1.


def test_seven_point_laplacian_neighbor_weights_match_sympy() -> None:
    """Each of the 6 face-neighbor weights must equal the SymPy coefficient."""
    from tests.test_kernels import seven_point_laplacian  # reuse existing op

    # SymPy: 2nd-order centered d²f/dx², coefficient at offset ±1 is 1.
    # Same coefficient applies in each of the 3 axis directions.
    expected_neighbor = float(fd_coefficients(2, [-1, 0, 1])[0])  # Rational(1)

    face_neighbors = [
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    ]
    weights = probe_operator_weights(seven_point_laplacian, face_neighbors)

    for offset, w in weights.items():
        assert w == pytest.approx(expected_neighbor), (
            f"Neighbor weight at offset {offset} is {w}; "
            f"expected {expected_neighbor} (SymPy)"
        )


def test_seven_point_laplacian_center_weight_matches_sympy() -> None:
    """Center weight must be 3 × (SymPy 1D center coefficient) = 3 × (−2) = −6."""
    from tests.test_kernels import seven_point_laplacian

    # 1D center coefficient from SymPy: -2.
    # 3D 7-point Laplacian sums three independent 1D stencils, so center = 3 × (-2).
    center_1d = float(fd_coefficients(2, [-1, 0, 1])[1])  # Rational(-2)
    expected_center = 3 * center_1d  # -6.0

    weights = probe_operator_weights(seven_point_laplacian, [(0, 0, 0)])
    assert weights[(0, 0, 0)] == pytest.approx(expected_center), (
        f"Center weight is {weights[(0, 0, 0)]}; "
        f"expected {expected_center} (3 × SymPy 1D center)"
    )
