"""Convergence-order measurement helpers for numerical scheme tests.

Usage pattern
-------------
Define an ``error_fn(n: int) -> float`` that runs your scheme at grid
resolution *n* and returns a scalar error norm (L1, L2, or L∞) against
a known exact solution, then call :func:`assert_convergence_order`::

    def laplacian_l2_error(n: int) -> float:
        ...  # run scheme, return ||numerical - exact||_2

    def test_laplacian_converges_second_order() -> None:
        # docstring must state: smooth IC, periodic BCs, conservative scheme
        assert_convergence_order(laplacian_l2_error, [16, 32, 64, 128], expected=2.0)

When convergence testing is appropriate
----------------------------------------
- Smooth initial data with a known exact or manufactured solution.
- Mesh operators (gradient, divergence, Laplacian): polynomial exact
  solutions exist for any order; SymPy can generate them.
- Time integrators in isolation: vary Δt, hold spatial error negligible.
- Method of Manufactured Solutions (MMS): add a source term so any smooth
  function satisfies the equation exactly; works when natural exact
  solutions are hard to find.

When convergence testing is NOT appropriate
--------------------------------------------
- Riemann problems / shocks: TVD and monotonicity constraints reduce the
  scheme to first order at discontinuities by design. Apply the scheme to
  smooth data instead, or use the exact Riemann solution at a single
  resolution.
- Limiter-active regimes: limiters deliberately reduce order near extrema.
  Test with data smooth enough that the limiter never fires.
- Spatial operator tests with an over-large Δt: time integration error
  contaminates the spatial convergence measurement. Use a steady-state
  problem or take Δt ~ h^(p+1) so the time error is sub-dominant.
- Non-conservative formulations: convergence order in conserved quantities
  is not guaranteed. Test what the scheme is actually designed to do.

Norm choice
-----------
Use the L2 norm by default.  For problems that include shocks (tested at
lower resolutions for other purposes), L1 sometimes converges at first
order even for higher-order schemes; document the choice in the test.

Resolution count
----------------
Three resolutions is the minimum; the log-log fit is noisy with three
points.  Five or more is preferred for a stable slope estimate.  Use
powers of two so that successive errors differ by approximately 2^p.

"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np


def measure_convergence_order(
    error_fn: Callable[[int], float],
    resolutions: Sequence[int],
) -> float:
    """Fit *p* in ``error ~ C·h^p`` via log-log regression.

    Parameters
    ----------
    error_fn:
        Maps grid size *N* to a scalar error norm.  Called once per
        resolution; the caller owns any setup and teardown.
    resolutions:
        Strictly increasing sequence of grid sizes, e.g.
        ``[16, 32, 64, 128]``.  At least three points required; five or
        more gives a stable fit.

    Returns
    -------
    float
        Measured convergence exponent *p*.
    """
    if len(resolutions) < 3:
        raise ValueError(
            f"At least 3 resolutions required for a stable fit; got {len(resolutions)}"
        )
    errors = [error_fn(n) for n in resolutions]
    log_h = np.log(1.0 / np.asarray(resolutions, dtype=float))
    log_e = np.log(np.asarray(errors, dtype=float))
    slope, _ = np.polyfit(log_h, log_e, 1)
    return float(slope)


def assert_convergence_order(
    error_fn: Callable[[int], float],
    resolutions: Sequence[int],
    expected: float,
    *,
    atol: float = 0.15,
) -> float:
    """Assert that the measured convergence order is within *atol* of *expected*.

    Parameters
    ----------
    error_fn:
        Maps grid size *N* to a scalar error norm.
    resolutions:
        Strictly increasing grid-size sequence; see :func:`measure_convergence_order`.
    expected:
        Design order of the scheme (e.g. ``2.0`` for second-order).
    atol:
        Acceptable deviation from *expected*.  Default 0.15 is appropriate
        for four resolutions; tighten to ~0.05 with six or more.

    Returns
    -------
    float
        Measured convergence order (useful for diagnostic printing on failure).

    Notes
    -----
    The test's docstring **must** document the validity conditions for the
    convergence assertion: boundary condition type, whether the scheme is
    conservative, and whether the initial data is smooth enough that
    limiters do not fire.  A convergence test without this documentation
    is treated as a review finding (see ``roadmap/epoch-02-mesh.md``).
    """
    errors = [error_fn(n) for n in resolutions]
    log_h = np.log(1.0 / np.asarray(resolutions, dtype=float))
    log_e = np.log(np.asarray(errors, dtype=float))
    slope, _ = np.polyfit(log_h, log_e, 1)
    measured = float(slope)
    assert abs(measured - expected) <= atol, (
        f"Convergence order {measured:.3f} differs from expected {expected:.1f} "
        f"by more than atol={atol}.\n"
        f"  resolutions : {list(resolutions)}\n"
        f"  errors      : {[f'{e:.3e}' for e in errors]}"
    )
    return measured
