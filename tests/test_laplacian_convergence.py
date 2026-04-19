"""Convergence test: second-order 7-point Laplacian against an analytical solution.

Applies the production ``seven_point_laplacian`` stencil to a smooth,
periodic field and verifies that the L2 error between the numerical
approximation and the exact Laplacian shrinks at second order as the
grid is refined.

Validity conditions
-------------------
- Smooth initial data: f(x,y,z) = sin(2πx) is analytic, so all
  derivatives exist and the Taylor-expansion truncation error is
  O(h²) as derived in ``derivations/laplacian_stencil.py``.
- Periodic boundary conditions: halos are filled by wrapping (jnp.pad
  with mode='wrap'), so there are no boundary-flux contributions.
- Linear operator: the 7-point stencil is a linear finite-difference
  operator; no limiters or non-linear flux reconstructions are active.
- Conservative: the stencil weights sum to zero, so a uniform field
  maps to zero exactly.
- Exact solution: ∇²f = -(2π)²sin(2πx).
"""

from __future__ import annotations

import jax.numpy as jnp

from cosmic_foundry.computation.array import Array
from cosmic_foundry.computation.descriptor import Extent
from cosmic_foundry.computation.stencil import seven_point_laplacian
from tests.utils.convergence import assert_convergence_order


def _laplacian_l2_error(n: int) -> float:
    """L2 error of the numerical Laplacian at grid size n×n×n.

    Grid: uniform [0, 1)³ with n points per axis, spacing h = 1/n.
    Field: φ(x, y, z) = sin(2πx).  Exact: ∇²φ = -(2π)²sin(2πx).
    Periodic halos are applied via jnp.pad(mode='wrap').
    """
    h = 1.0 / n
    x = jnp.linspace(0.0, 1.0, n, endpoint=False)
    phi = jnp.sin(2.0 * jnp.pi * x)[:, None, None] * jnp.ones((n, n, n))

    # Periodic halo of width 1 in all directions.
    phi_padded = jnp.pad(phi, 1, mode="wrap")

    # Interior extent within the padded array: indices 1..n in each axis.
    extent = Extent((slice(1, n + 1), slice(1, n + 1), slice(1, n + 1)))
    result = seven_point_laplacian.execute(Array((phi_padded,)), extent=extent)

    # Stencil returns the un-divided form; divide by h² to get ∇²φ.
    numerical = result / h**2

    exact = -((2.0 * jnp.pi) ** 2) * phi
    error = float(jnp.sqrt(jnp.mean((numerical - exact) ** 2)))
    return error


def test_laplacian_converges_second_order() -> None:
    """7-point Laplacian on sin(2πx) achieves second-order convergence."""
    assert_convergence_order(
        _laplacian_l2_error,
        [16, 32, 64, 128, 256],
        expected=2.0,
        atol=0.05,
    )
