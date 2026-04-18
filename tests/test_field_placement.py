"""Tests for Placement, covers(), and the Field → Op integration.

Placement unit tests verify the ComponentId→rank mapping API.
covers() tests verify that Array[Patch] correctly reports spatial coverage.
The integration tests run the seven-point Laplacian on φ = x²+y²+z² and
verify the expected result (6.0) in both single-process and multi-process modes.
"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import pytest

from cosmic_foundry.descriptor import AccessPattern, Extent, Region
from cosmic_foundry.function import Function, execute_pointwise
from cosmic_foundry.mesh import covers, partition_domain
from cosmic_foundry.record import ComponentId, Placement

N = 8


@dataclass(frozen=True)
class SevenPointLaplacian(Function):
    """Seven-point finite-difference Laplacian on a 3-D grid.

    Function:
        domain   — φ: DiscreteField on Ω_h ⊆ ℝ³
        codomain — ∇²φ: DiscreteField on Ω_h^int ⊆ Ω_h
        operator — (∇²φ)_{ijk} = φ_{i-1,jk} + φ_{i+1,jk} + φ_{i,j-1,k}
                                + φ_{i,j+1,k} + φ_{ij,k-1} + φ_{ij,k+1}
                                - 6 φ_{ijk}

    Θ = {h}, p = 2 — second-order finite-difference approximation of ∇².
    Exact for polynomials of degree ≤ 2.
    """

    @property
    def access_pattern(self) -> AccessPattern:
        return AccessPattern.seven_point()

    def execute(self, phi: Any, *, region: Region) -> Any:
        return execute_pointwise(self, region, phi)

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


@pytest.fixture()
def phi() -> jnp.ndarray:
    axes = jnp.indices((N, N, N), dtype=jnp.float64)
    return axes[0] ** 2 + axes[1] ** 2 + axes[2] ** 2


# ---------------------------------------------------------------------------
# Placement unit tests
# ---------------------------------------------------------------------------


def test_placement_owner_lookup() -> None:
    p = Placement({ComponentId(0): 0, ComponentId(1): 1})
    assert p.owner(ComponentId(0)) == 0
    assert p.owner(ComponentId(1)) == 1


def test_placement_segments_for_rank() -> None:
    p = Placement({ComponentId(0): 0, ComponentId(1): 0, ComponentId(2): 1})
    assert p.segments_for_rank(0) == {ComponentId(0), ComponentId(1)}
    assert p.segments_for_rank(1) == {ComponentId(2)}
    assert p.segments_for_rank(99) == frozenset()


def test_placement_rejects_unknown_segment() -> None:
    p = Placement({ComponentId(0): 0})
    with pytest.raises(KeyError):
        p.owner(ComponentId(99))


def test_placement_rejects_negative_rank() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        Placement({ComponentId(0): -1})


def test_placement_rejects_empty() -> None:
    with pytest.raises(ValueError, match="at least one"):
        Placement({})


# ---------------------------------------------------------------------------
# covers() tests
# ---------------------------------------------------------------------------


def test_covers_single_block_full_extent() -> None:
    mesh = partition_domain(
        domain_origin=(0.0, 0.0, 0.0),
        domain_size=(float(N), float(N), float(N)),
        n_cells=(N, N, N),
        blocks_per_axis=(1, 1, 1),
        n_ranks=1,
    )
    full = Extent.from_shape((N, N, N))
    assert covers(mesh, full)


def test_covers_two_blocks_cover_split_domain() -> None:
    mesh = partition_domain(
        domain_origin=(0.0, 0.0, 0.0),
        domain_size=(float(N), float(N), float(N)),
        n_cells=(N, N, N),
        blocks_per_axis=(2, 1, 1),
        n_ranks=1,
    )
    assert covers(mesh, Extent.from_shape((N, N, N)))


def test_covers_rejects_extent_outside_mesh() -> None:
    """A mesh covering [0, N) does not cover an extent that exceeds that range."""
    mesh = partition_domain(
        domain_origin=(0.0, 0.0, 0.0),
        domain_size=(float(N), float(N), float(N)),
        n_cells=(N, N, N),
        blocks_per_axis=(1, 1, 1),
        n_ranks=1,
    )
    # Expand [0, N)^3 by 1 → [-1, N+1)^3 which the mesh cannot cover
    full = Extent.from_shape((N, N, N))
    beyond = full.expand(seven_point_laplacian.access_pattern)
    assert not covers(mesh, beyond)


# ---------------------------------------------------------------------------
# Single-process integration
# ---------------------------------------------------------------------------


def test_single_process_field_op_laplacian(phi: jnp.ndarray) -> None:
    """One block, full domain — the degenerate single-rank case."""
    mesh = partition_domain(
        domain_origin=(0.0, 0.0, 0.0),
        domain_size=(float(N), float(N), float(N)),
        n_cells=(N, N, N),
        blocks_per_axis=(1, 1, 1),
        n_ranks=1,
    )
    full = Extent.from_shape((N, N, N))
    required = full.expand(seven_point_laplacian.access_pattern)
    assert not covers(mesh, required)  # mesh doesn't cover the halo ring

    interior = Extent((slice(1, N - 1), slice(1, N - 1), slice(1, N - 1)))
    assert covers(mesh, interior)  # mesh does cover the interior

    result = seven_point_laplacian.execute(phi, region=Region(interior))
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
