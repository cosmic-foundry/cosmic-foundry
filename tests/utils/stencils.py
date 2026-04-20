"""SymPy-based stencil verification utilities.

``fd_coefficients`` derives exact rational finite-difference coefficients
from first principles via Taylor-expansion, returning ``sympy.Rational``
objects so comparisons are exact.

Notes on output units
---------------------
The returned coefficients are for the *divided* form:

    f^(n)(x) ≈ Σᵢ cᵢ · f(x₀ + offsets[i]·h) / h^n

For a kernel that returns the *un-divided* stencil (omitting the h^n
denominator), multiply each coefficient by h^n before comparison, or test
on a unit-spacing grid (h=1) where divided and un-divided are equivalent.

See also: ``tests/utils/convergence.py`` — complementary convergence-order helper.
"""

from __future__ import annotations

from collections.abc import Sequence

from sympy import Rational
from sympy.calculus.finite_diff import finite_diff_weights


def fd_coefficients(
    deriv_order: int,
    offsets: Sequence[int],
) -> list[Rational]:
    """Return exact rational FD coefficients for a given stencil.

    Uses Taylor-expansion to solve for the unique set of coefficients
    that approximates ``f^(deriv_order)`` to the highest achievable order
    given the supplied offsets.

    Parameters
    ----------
    deriv_order:
        Which derivative to approximate (1 = first, 2 = second, …).
    offsets:
        Integer offsets from the stencil center, e.g. ``[-1, 0, 1]``
        for a three-point stencil.  Offsets are in units of grid spacing h.

    Returns
    -------
    list of sympy.Rational
        Coefficients cᵢ such that
        ``f^(n)(x) ≈ Σᵢ cᵢ · f(x + offsets[i]·h) / h^n``.
        The i-th element corresponds to ``offsets[i]``.

    Examples
    --------
    Second-order centered second derivative (3-point):

    >>> fd_coefficients(2, [-1, 0, 1])
    [1, -2, 1]

    Fourth-order centered second derivative (5-point):

    >>> fd_coefficients(2, [-2, -1, 0, 1, 2])
    [-1/12, 4/3, -5/2, 4/3, -1/12]

    Second-order centered first derivative (3-point):

    >>> fd_coefficients(1, [-1, 0, 1])
    [-1/2, 0, 1/2]
    """
    if deriv_order < 1:
        raise ValueError(f"deriv_order must be >= 1; got {deriv_order}")
    offsets_list = list(offsets)
    if len(offsets_list) < deriv_order + 1:
        raise ValueError(
            f"Need at least deriv_order+1 = {deriv_order+1} offsets "
            f"to approximate the {deriv_order}-th derivative; "
            f"got {len(offsets_list)}"
        )
    # finite_diff_weights returns weights[d][n] where d indexes derivative
    # order and n indexes number of points used (0-based).  We want the
    # coefficients for exactly deriv_order using all supplied points:
    # weights[deriv_order][-1].
    weights = finite_diff_weights(deriv_order, offsets_list, 0)
    return [Rational(w) for w in weights[deriv_order][-1]]
