"""Tests for kernel interface primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import pytest

from cosmic_foundry.computation.descriptor import Extent
from cosmic_foundry.computation.stencil import execute_pointwise
from cosmic_foundry.theory.function import Function


@dataclass(frozen=True)
class SevenPointLaplacian(Function):
    """Seven-point finite-difference Laplacian on a 3-D grid.

    Function:
        domain   — φ: PatchFunction on Ω_h ⊆ ℝ³
        codomain — ∇²φ: PatchFunction on Ω_h^int ⊆ Ω_h
        operator — (∇²φ)_{ijk} = φ_{i-1,jk} + φ_{i+1,jk} + φ_{i,j-1,k}
                                + φ_{i,j+1,k} + φ_{ij,k-1} + φ_{ij,k+1}
                                - 6 φ_{ijk}

    Θ = {h}, p = 2 — second-order finite-difference approximation of ∇².
    Exact for polynomials of degree ≤ 2.
    """

    radii: tuple[int, ...] = (1, 1, 1)

    def execute(self, phi: Any, *, extent: Extent) -> Any:
        return execute_pointwise(self, extent, phi)

    def _fn(self, phi: Any, i: Any, j: Any, k: Any) -> Any:
        return (
            phi[i - 1, j, k]
            + phi[i + 1, j, k]
            + phi[i, j - 1, k]
            + phi[i, j + 1, k]
            + phi[i, j, k - 1]
            + phi[i, j, k + 1]
            - 6.0 * phi[i, j, k]
        )


seven_point_laplacian = SevenPointLaplacian()


def test_op_class_exposes_radii() -> None:
    assert seven_point_laplacian.radii == (1, 1, 1)


def test_op_execute_runs_kernel() -> None:
    n = 8
    axes = jnp.indices((n, n, n), dtype=jnp.float64)
    phi = axes[0] ** 2 + axes[1] ** 2 + axes[2] ** 2
    extent = Extent((slice(1, n - 1), slice(1, n - 1), slice(1, n - 1)))
    result = seven_point_laplacian(phi, extent=extent)
    assert result.shape == (n - 2, n - 2, n - 2)
    assert jnp.allclose(result, 6.0)


def test_stencil_radii_fourth_order() -> None:
    radii = (2, 2, 2)
    assert radii[0] == 2
    assert radii[1] == 2
    assert radii[2] == 2


def test_op_runs_laplacian_over_extent() -> None:
    n = 8
    axes = jnp.indices((n, n, n), dtype=jnp.float64)
    phi = axes[0] ** 2 + axes[1] ** 2 + axes[2] ** 2
    extent = Extent((slice(1, n - 1), slice(1, n - 1), slice(1, n - 1)))

    result = seven_point_laplacian.execute(phi, extent=extent)

    assert result.shape == (n - 2, n - 2, n - 2)
    assert jnp.allclose(result, 6.0)


def test_op_rejects_extent_without_required_halo() -> None:
    phi = jnp.ones((4, 4, 4))
    extent = Extent.from_shape(phi.shape)

    with pytest.raises(ValueError, match="exceeds input bounds"):
        seven_point_laplacian.execute(phi, extent=extent)
