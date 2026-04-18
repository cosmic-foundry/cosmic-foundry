"""Tests for the Field placement data model.

Non-distributed tests cover the ComponentId / DiscreteField / Placement / Field
API and the Field.covers() validation.

The multi-rank test at the bottom spawns two subprocesses that initialize
``jax.distributed`` and run the Laplacian on disjoint half-domains, then
checks that every rank returned 6.0.  It runs in the standard test suite;
no special flags are required.
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

from cosmic_foundry.fields import DiscreteField, Placement
from cosmic_foundry.kernels import (
    AccessPattern,
    ComponentId,
    Extent,
    Map,
    Region,
    execute_pointwise,
)

# ---------------------------------------------------------------------------
# Shared Op (mirrors test_kernels.py; defined here to keep tests independent)
# ---------------------------------------------------------------------------

N = 8


@dataclass(frozen=True)
class SevenPointLaplacian(Map):
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
# Field unit tests
# ---------------------------------------------------------------------------


def test_field_rejects_segment_not_in_placement(phi: jnp.ndarray) -> None:
    seg = DiscreteField(
        name="phi",
        segment_id=ComponentId(99),
        payload=phi,
        extent=Extent.from_shape(phi.shape),
    )
    with pytest.raises(ValueError, match="not registered"):
        DiscreteField(
            name="phi", segments=(seg,), placement=Placement({ComponentId(0): 0})
        )


def test_field_segment_rejects_interior_outside_extent(phi: jnp.ndarray) -> None:
    with pytest.raises(ValueError, match="interior_extent"):
        DiscreteField(
            name="phi",
            segment_id=ComponentId(0),
            payload=phi,
            extent=Extent((slice(0, 4), slice(0, N), slice(0, N))),
            interior_extent=Extent((slice(3, 5), slice(0, N), slice(0, N))),
        )


def test_field_local_segments_single_process(phi: jnp.ndarray) -> None:
    seg = DiscreteField(
        name="phi",
        segment_id=ComponentId(0),
        payload=phi,
        extent=Extent.from_shape(phi.shape),
    )
    field = DiscreteField(
        name="phi", segments=(seg,), placement=Placement({ComponentId(0): 0})
    )
    assert field.local_segments(0) == (seg,)
    assert field.local_segments(1) == ()


def test_field_segment_lookup(phi: jnp.ndarray) -> None:
    seg = DiscreteField(
        name="phi",
        segment_id=ComponentId(0),
        payload=phi,
        extent=Extent.from_shape(phi.shape),
    )
    field = DiscreteField(
        name="phi", segments=(seg,), placement=Placement({ComponentId(0): 0})
    )
    assert field.segment(ComponentId(0)) is seg
    with pytest.raises(KeyError):
        field.segment(ComponentId(99))


# ---------------------------------------------------------------------------
# Field.covers() tests
# ---------------------------------------------------------------------------


def test_single_segment_covers_full_extent(phi: jnp.ndarray) -> None:
    full = Extent.from_shape(phi.shape)
    seg = DiscreteField(name="phi", segment_id=ComponentId(0), payload=phi, extent=full)
    field = DiscreteField(
        name="phi", segments=(seg,), placement=Placement({ComponentId(0): 0})
    )
    assert field.covers(full)


def test_two_disjoint_segments_cover_split_domain(phi: jnp.ndarray) -> None:
    half = N // 2
    ext0 = Extent((slice(0, half), slice(0, N), slice(0, N)))
    ext1 = Extent((slice(half, N), slice(0, N), slice(0, N)))
    seg0 = DiscreteField(
        name="phi", segment_id=ComponentId(0), payload=phi[:half], extent=ext0
    )
    seg1 = DiscreteField(
        name="phi", segment_id=ComponentId(1), payload=phi[half:], extent=ext1
    )
    field = DiscreteField(
        name="phi",
        segments=(seg0, seg1),
        placement=Placement({ComponentId(0): 0, ComponentId(1): 1}),
    )
    assert field.covers(Extent.from_shape(phi.shape))


def test_gap_in_coverage_is_detected(phi: jnp.ndarray) -> None:
    # Row 3 is missing between the two segments.
    ext0 = Extent((slice(0, 3), slice(0, N), slice(0, N)))
    ext1 = Extent((slice(4, N), slice(0, N), slice(0, N)))
    seg0 = DiscreteField(
        name="phi", segment_id=ComponentId(0), payload=phi[:3], extent=ext0
    )
    seg1 = DiscreteField(
        name="phi", segment_id=ComponentId(1), payload=phi[4:], extent=ext1
    )
    field = DiscreteField(
        name="phi",
        segments=(seg0, seg1),
        placement=Placement({ComponentId(0): 0, ComponentId(1): 1}),
    )
    assert not field.covers(Extent.from_shape(phi.shape))


def test_covers_checks_halo_expansion(phi: jnp.ndarray) -> None:
    # A segment that covers only the interior is not sufficient once the
    # 7-point stencil halo is added.
    interior = Extent((slice(1, N - 1), slice(1, N - 1), slice(1, N - 1)))
    seg = DiscreteField(
        name="phi",
        segment_id=ComponentId(0),
        payload=phi[1 : N - 1, 1 : N - 1, 1 : N - 1],
        extent=interior,
    )
    field = DiscreteField(
        name="phi", segments=(seg,), placement=Placement({ComponentId(0): 0})
    )
    required = interior.expand(seven_point_laplacian.access_pattern)
    assert not field.covers(required)


# ---------------------------------------------------------------------------
# Single-process Field → Op integration (degenerate case)
# ---------------------------------------------------------------------------


def test_single_process_field_op_laplacian(phi: jnp.ndarray) -> None:
    """One Field, one segment, full domain — the degenerate single-rank case."""
    full = Extent.from_shape(phi.shape)
    seg = DiscreteField(name="phi", segment_id=ComponentId(0), payload=phi, extent=full)
    field = DiscreteField(
        name="phi", segments=(seg,), placement=Placement({ComponentId(0): 0})
    )

    interior = Extent((slice(1, N - 1), slice(1, N - 1), slice(1, N - 1)))
    required = interior.expand(seven_point_laplacian.access_pattern)
    assert field.covers(required)

    result = seven_point_laplacian.execute(seg.payload, region=Region(interior))
    assert jnp.allclose(result, 6.0)


# ---------------------------------------------------------------------------
# Multi-rank correctness harness (requires --multihost)
# ---------------------------------------------------------------------------


def test_multi_rank_field_placement_laplacian() -> None:
    """Two ranks each compute the Laplacian on their partition via jax.distributed.

    Rank 0 owns rows [0, half+1); rank 1 owns rows [half-1, n).  Each covers
    a disjoint interior [1, half) in local coordinates.  All results must be
    6.0, matching the single-rank computation on phi = x^2 + y^2 + z^2.
    """
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
