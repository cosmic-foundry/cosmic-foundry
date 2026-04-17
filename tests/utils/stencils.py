"""SymPy-based stencil verification utilities.

Design intent
-------------
SymPy is used here as a **verification oracle** — it derives exact rational
finite-difference coefficients from first principles so that test code can
compare them against empirically measured kernel weights.  SymPy is *not*
used in simulation hot paths; the production kernels carry only plain float
arrays.

Two complementary tools are provided:

``fd_coefficients``
    Pure-SymPy: derives the exact rational FD coefficients for any
    derivative order and stencil offset list via Taylor-expansion.
    Returns ``sympy.Rational`` objects so comparisons are exact.

``probe_operator_weights``
    JAX-side: applies a kernel to unit-impulse inputs to empirically
    extract the weight assigned to each stencil offset.  Works on any
    linear kernel without requiring it to expose its coefficients
    explicitly.

The combination catches coefficient errors that happen to cancel on
simple smooth test functions (e.g. the x²+y²+z² Laplacian test will
pass even if the center weight is -5 instead of -6, because the
quadratic polynomial has zero fourth-order terms — probing with unit
impulses catches this directly).

Notes on ``fd_coefficients`` output units
------------------------------------------
The returned coefficients are for the *divided* form:

    f^(n)(x) ≈ Σᵢ cᵢ · f(x₀ + offsets[i]·h) / h^n

For a kernel that returns the *un-divided* stencil (omitting the h^n
denominator, as ``seven_point_laplacian`` does), multiply each coefficient
by h^n before comparison, or test on a unit-spacing grid (h=1) where
divided and un-divided are equivalent.

See also
--------
``replication/formulas.md`` — formula register where every stencil's
source paper and order are recorded.
``tests/utils/convergence.py`` — complementary convergence-order helper.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
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


def probe_operator_weights(
    kernel_fn,
    offsets_3d: Sequence[tuple[int, int, int]],
    *,
    n: int = 7,
) -> dict[tuple[int, int, int], float]:
    """Empirically extract the weight a linear kernel assigns to each offset.

    For each offset in ``offsets_3d``, constructs a JAX array that is 1.0 at
    center + offset and 0.0 everywhere else, applies ``kernel_fn`` at the
    center index, and records the output.  For a linear kernel this output is
    exactly the stencil weight for that offset.

    Parameters
    ----------
    kernel_fn:
        A pointwise kernel callable with signature
        ``kernel_fn(phi, i, j, k) -> float``.  Must be linear in phi.
    offsets_3d:
        Sequence of (di, dj, dk) tuples to probe.
    n:
        Side length of the probe array.  Must be large enough that
        ``center ± max(|offset|)`` stays in-bounds.

    Returns
    -------
    dict mapping (di, dj, dk) -> measured weight

    Notes
    -----
    This function calls ``kernel_fn`` directly at the center index using
    a plain NumPy-backed probe array — it does not go through
    ``Dispatch`` / ``Region``.  This keeps the probe independent of the
    dispatch machinery and works for any kernel with the pointwise
    signature.
    """
    center = n // 2
    weights: dict[tuple[int, int, int], float] = {}

    for offset in offsets_3d:
        di, dj, dk = offset
        phi = jnp.zeros((n, n, n), dtype=jnp.float64)
        phi = phi.at[center + di, center + dj, center + dk].set(1.0)
        result = kernel_fn(phi, center, center, center)
        weights[offset] = float(result)

    return weights
