"""Tests for kernel interface primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import jax
import jax.numpy as jnp
import pytest

from cosmic_foundry.kernels import (
    AccessPattern,
    Extent,
    Op,
    Region,
)


@dataclass(frozen=True)
class SevenPointLaplacian(Op):
    """Seven-point finite-difference Laplacian on a 3-D grid.

    Map:
        domain   — φ: DiscreteField on Ω_h ⊆ ℝ³
        codomain — ∇²φ: DiscreteField on Ω_h^int ⊆ Ω_h
        operator — (∇²φ)_{ijk} = φ_{i-1,jk} + φ_{i+1,jk} + φ_{i,j-1,k}
                                + φ_{i,j+1,k} + φ_{ij,k-1} + φ_{ij,k+1}
                                - 6 φ_{ijk}

    Θ = {h}, p = 2 — second-order finite-difference approximation of ∇².
    Exact for polynomials of degree ≤ 2.
    """

    reads: ClassVar[tuple[str, ...]] = ("phi",)
    writes: ClassVar[tuple[str, ...]] = ("laplacian_phi",)

    @property
    def access_pattern(self) -> AccessPattern:
        return AccessPattern.seven_point()

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


def test_op_class_exposes_metadata() -> None:
    assert seven_point_laplacian.access_pattern == AccessPattern.seven_point()
    assert seven_point_laplacian.reads == ("phi",)
    assert seven_point_laplacian.writes == ("laplacian_phi",)


def test_op_execute_runs_kernel() -> None:
    n = 8
    axes = jnp.indices((n, n, n), dtype=jnp.float64)
    phi = axes[0] ** 2 + axes[1] ** 2 + axes[2] ** 2
    region = Region(Extent((slice(1, n - 1), slice(1, n - 1), slice(1, n - 1))))
    result = seven_point_laplacian(phi, region=region)
    assert result.shape == (n - 2, n - 2, n - 2)
    assert jnp.allclose(result, 6.0)


def test_stencil_symmetric_sets_halo_width() -> None:
    stencil = AccessPattern.symmetric(order=4, ndim=3)
    assert stencil.halo_width(0) == 2
    assert stencil.halo_width(1) == 2
    assert stencil.halo_width(2) == 2


def test_op_runs_laplacian_over_region() -> None:
    n = 8
    axes = jnp.indices((n, n, n), dtype=jnp.float64)
    phi = axes[0] ** 2 + axes[1] ** 2 + axes[2] ** 2
    region = Region(Extent((slice(1, n - 1), slice(1, n - 1), slice(1, n - 1))))

    result = seven_point_laplacian.execute(phi, region=region)

    assert result.shape == (n - 2, n - 2, n - 2)
    assert jnp.allclose(result, 6.0)


def test_op_rejects_region_without_required_halo() -> None:
    phi = jnp.ones((4, 4, 4))
    region = Region(Extent.from_shape(phi.shape))

    with pytest.raises(ValueError, match="exceeds input bounds"):
        seven_point_laplacian.execute(phi, region=region)


def test_op_runs_laplacian_over_batched_region() -> None:
    """Batched region: same Op runs over n_blocks stacked blocks via vmap."""
    n = 8
    n_blocks = 3
    axes = jnp.indices((n, n, n), dtype=jnp.float64)
    phi_single = axes[0] ** 2 + axes[1] ** 2 + axes[2] ** 2
    phi_batched = jnp.stack([phi_single] * n_blocks)  # (3, 8, 8, 8)

    region = Region(
        Extent((slice(1, n - 1), slice(1, n - 1), slice(1, n - 1))),
        n_blocks=n_blocks,
    )
    result = seven_point_laplacian.execute(phi_batched, region=region)

    assert result.shape == (n_blocks, n - 2, n - 2, n - 2)
    assert jnp.allclose(result, 6.0)


def test_batched_region_matches_single_block_results() -> None:
    """Each batched-region slice must equal the corresponding single-block result."""
    n = 6

    def make_phi(offset: float) -> jax.Array:
        axes = jnp.indices((n, n, n), dtype=jnp.float64)
        return axes[0] ** 2 + axes[1] ** 2 + axes[2] ** 2 + offset

    blocks = [make_phi(float(k)) for k in range(4)]
    extent = Extent((slice(1, n - 1), slice(1, n - 1), slice(1, n - 1)))

    single_results = [
        seven_point_laplacian.execute(phi, region=Region(extent)) for phi in blocks
    ]

    batched_result = seven_point_laplacian.execute(
        jnp.stack(blocks),
        region=Region(extent, n_blocks=len(blocks)),
    )

    for i, expected in enumerate(single_results):
        assert jnp.allclose(batched_result[i], expected), f"block {i} mismatch"


def test_batched_region_rejects_mismatched_batch_size() -> None:
    n = 6
    phi_batched = jnp.ones((5, n, n, n))
    region = Region(
        Extent((slice(1, n - 1), slice(1, n - 1), slice(1, n - 1))),
        n_blocks=3,
    )
    with pytest.raises(ValueError, match="n_blocks"):
        seven_point_laplacian.execute(phi_batched, region=region)


def test_region_rejects_nonpositive_n_blocks() -> None:
    with pytest.raises(ValueError, match="n_blocks"):
        Region(Extent.from_shape((4, 4, 4)), n_blocks=0)
