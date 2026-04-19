"""Tests for covers() and the Field → Op integration."""

from __future__ import annotations

import json
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import pytest

from cosmic_foundry.computation.array import Array
from cosmic_foundry.computation.descriptor import Extent
from cosmic_foundry.computation.stencil import Stencil
from cosmic_foundry.geometry.domain import Domain
from cosmic_foundry.geometry.euclidean_space import EuclideanSpace
from cosmic_foundry.mesh import covers, partition_domain

N = 8


def _seven_point_fn(fields: tuple[Any, ...], i: Any, j: Any, k: Any) -> Any:
    phi = fields[0]
    return (
        phi[i - 1, j, k]
        + phi[i + 1, j, k]
        + phi[i, j - 1, k]
        + phi[i, j + 1, k]
        + phi[i, j, k - 1]
        + phi[i, j, k + 1]
        - 6.0 * phi[i, j, k]
    )


seven_point_laplacian = Stencil(fn=_seven_point_fn, radii=(1, 1, 1))


@pytest.fixture()
def phi() -> jnp.ndarray:
    axes = jnp.indices((N, N, N), dtype=jnp.float64)
    return axes[0] ** 2 + axes[1] ** 2 + axes[2] ** 2


# ---------------------------------------------------------------------------
# covers() tests
# ---------------------------------------------------------------------------


def test_covers_single_block_full_extent() -> None:
    mesh = partition_domain.execute(
        domain=Domain(
            manifold=EuclideanSpace(3),
            origin=(0.0, 0.0, 0.0),
            size=(float(N), float(N), float(N)),
        ),
        n_cells=(N, N, N),
        blocks_per_axis=(1, 1, 1),
    )
    assert covers(mesh, Extent.from_shape((N, N, N)))


def test_covers_two_blocks_cover_split_domain() -> None:
    mesh = partition_domain.execute(
        domain=Domain(
            manifold=EuclideanSpace(3),
            origin=(0.0, 0.0, 0.0),
            size=(float(N), float(N), float(N)),
        ),
        n_cells=(N, N, N),
        blocks_per_axis=(2, 1, 1),
    )
    assert covers(mesh, Extent.from_shape((N, N, N)))


def test_covers_rejects_extent_outside_mesh() -> None:
    mesh = partition_domain.execute(
        domain=Domain(
            manifold=EuclideanSpace(3),
            origin=(0.0, 0.0, 0.0),
            size=(float(N), float(N), float(N)),
        ),
        n_cells=(N, N, N),
        blocks_per_axis=(1, 1, 1),
    )
    beyond = Extent.from_shape((N, N, N)).expand(seven_point_laplacian.radii)
    assert not covers(mesh, beyond)


# ---------------------------------------------------------------------------
# Single-process integration
# ---------------------------------------------------------------------------


def test_single_process_field_op_laplacian(phi: jnp.ndarray) -> None:
    mesh = partition_domain.execute(
        domain=Domain(
            manifold=EuclideanSpace(3),
            origin=(0.0, 0.0, 0.0),
            size=(float(N), float(N), float(N)),
        ),
        n_cells=(N, N, N),
        blocks_per_axis=(1, 1, 1),
    )
    interior = Extent((slice(1, N - 1), slice(1, N - 1), slice(1, N - 1)))
    assert covers(mesh, interior)

    result = seven_point_laplacian.execute(Array((phi,)), extent=interior)
    assert jnp.allclose(result, 6.0)


# ---------------------------------------------------------------------------
# Multi-rank correctness harness
# ---------------------------------------------------------------------------


def test_multi_rank_field_placement_laplacian() -> None:
    """Two ranks each compute the Laplacian on their partition via jax.distributed."""
    with socket.socket() as s:
        s.bind(("", 0))
        port = s.getsockname()[1]
    coordinator = f"localhost:{port}"

    worker = str(Path(__file__).parent / "_field_placement_worker.py")
    procs = [
        subprocess.Popen(
            [sys.executable, worker, str(rank), "2", coordinator],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for rank in range(2)
    ]

    results = []
    for proc in procs:
        stdout, stderr = proc.communicate(timeout=60)
        assert (
            proc.returncode == 0
        ), f"Worker exited {proc.returncode}:\n{stderr.decode()}"
        results.append(json.loads(stdout.decode().strip()))

    for r in results:
        assert r["ok"], f"Rank {r['rank']} reported error: {r.get('error')}"
        assert r["all_close_6"], f"Rank {r['rank']} Laplacian not close to 6.0"
