"""Tests for the Epoch 1 kernel interface nucleus."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from cosmic_foundry.kernels import Dispatch, Extent, Region, Stencil, op


@op(
    access_pattern=Stencil.seven_point(),
    reads=("phi",),
    writes=("laplacian_phi",),
)
def seven_point_laplacian(phi, i, j, k):
    return (
        phi[i - 1, j, k]
        + phi[i + 1, j, k]
        + phi[i, j - 1, k]
        + phi[i, j + 1, k]
        + phi[i, j, k - 1]
        + phi[i, j, k + 1]
        - 6.0 * phi[i, j, k]
    )


def test_op_decorator_attaches_metadata() -> None:
    assert seven_point_laplacian.access_pattern == Stencil.seven_point()
    assert seven_point_laplacian.reads == ("phi",)
    assert seven_point_laplacian.writes == ("laplacian_phi",)


def test_stencil_symmetric_sets_halo_width() -> None:
    stencil = Stencil.symmetric(order=4, ndim=3)
    assert stencil.halo_width(0) == 2
    assert stencil.halo_width(1) == 2
    assert stencil.halo_width(2) == 2


def test_dispatch_executes_laplacian_over_region() -> None:
    n = 8
    axes = jnp.indices((n, n, n), dtype=jnp.float64)
    phi = axes[0] ** 2 + axes[1] ** 2 + axes[2] ** 2
    region = Region(Extent((slice(1, n - 1), slice(1, n - 1), slice(1, n - 1))))

    result = Dispatch(
        seven_point_laplacian,
        region,
        inputs=(phi,),
    ).execute()

    assert result.shape == (n - 2, n - 2, n - 2)
    assert jnp.allclose(result, 6.0)


def test_dispatch_rejects_region_without_required_halo() -> None:
    phi = jnp.ones((4, 4, 4))
    region = Region(Extent.from_shape(phi.shape))

    with pytest.raises(ValueError, match="exceeds input bounds"):
        Dispatch(seven_point_laplacian, region, inputs=(phi,)).execute()
