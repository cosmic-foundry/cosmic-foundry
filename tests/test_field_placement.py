"""Tests for the Field placement data model.

Non-distributed tests cover the SegmentId / FieldSegment / Placement / Field
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
from pathlib import Path

import jax.numpy as jnp
import pytest

from cosmic_foundry.fields import Field, FieldSegment, Placement, SegmentId
from cosmic_foundry.kernels import Dispatch, Extent, Region, Stencil, op

# ---------------------------------------------------------------------------
# Shared Op (mirrors test_kernels.py; defined here to keep tests independent)
# ---------------------------------------------------------------------------

N = 8


@op(
    access_pattern=Stencil.seven_point(),
    reads=("phi",),
    writes=("laplacian_phi",),
)
def seven_point_laplacian(phi, i, j, k):  # type: ignore[no-untyped-def]
    return (
        phi[i - 1, j, k]
        + phi[i + 1, j, k]
        + phi[i, j - 1, k]
        + phi[i, j + 1, k]
        + phi[i, j, k - 1]
        + phi[i, j, k + 1]
        - 6.0 * phi[i, j, k]
    )


@pytest.fixture()
def phi() -> jnp.ndarray:
    axes = jnp.indices((N, N, N), dtype=jnp.float64)
    return axes[0] ** 2 + axes[1] ** 2 + axes[2] ** 2


# ---------------------------------------------------------------------------
# Placement unit tests
# ---------------------------------------------------------------------------


def test_placement_owner_lookup() -> None:
    p = Placement({SegmentId(0): 0, SegmentId(1): 1})
    assert p.owner(SegmentId(0)) == 0
    assert p.owner(SegmentId(1)) == 1


def test_placement_segments_for_rank() -> None:
    p = Placement({SegmentId(0): 0, SegmentId(1): 0, SegmentId(2): 1})
    assert p.segments_for_rank(0) == {SegmentId(0), SegmentId(1)}
    assert p.segments_for_rank(1) == {SegmentId(2)}
    assert p.segments_for_rank(99) == frozenset()


def test_placement_rejects_unknown_segment() -> None:
    p = Placement({SegmentId(0): 0})
    with pytest.raises(KeyError):
        p.owner(SegmentId(99))


def test_placement_rejects_negative_rank() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        Placement({SegmentId(0): -1})


def test_placement_rejects_empty() -> None:
    with pytest.raises(ValueError, match="at least one"):
        Placement({})


# ---------------------------------------------------------------------------
# Field unit tests
# ---------------------------------------------------------------------------


def test_field_rejects_segment_not_in_placement(phi: jnp.ndarray) -> None:
    seg = FieldSegment(SegmentId(99), phi, Extent.from_shape(phi.shape))
    with pytest.raises(ValueError, match="not registered"):
        Field("phi", (seg,), Placement({SegmentId(0): 0}))


def test_field_local_segments_single_process(phi: jnp.ndarray) -> None:
    seg = FieldSegment(SegmentId(0), phi, Extent.from_shape(phi.shape))
    field = Field("phi", (seg,), Placement({SegmentId(0): 0}))
    assert field.local_segments(0) == (seg,)
    assert field.local_segments(1) == ()


def test_field_segment_lookup(phi: jnp.ndarray) -> None:
    seg = FieldSegment(SegmentId(0), phi, Extent.from_shape(phi.shape))
    field = Field("phi", (seg,), Placement({SegmentId(0): 0}))
    assert field.segment(SegmentId(0)) is seg
    with pytest.raises(KeyError):
        field.segment(SegmentId(99))


# ---------------------------------------------------------------------------
# Field.covers() tests
# ---------------------------------------------------------------------------


def test_single_segment_covers_full_extent(phi: jnp.ndarray) -> None:
    full = Extent.from_shape(phi.shape)
    seg = FieldSegment(SegmentId(0), phi, full)
    field = Field("phi", (seg,), Placement({SegmentId(0): 0}))
    assert field.covers(full)


def test_two_disjoint_segments_cover_split_domain(phi: jnp.ndarray) -> None:
    half = N // 2
    ext0 = Extent((slice(0, half), slice(0, N), slice(0, N)))
    ext1 = Extent((slice(half, N), slice(0, N), slice(0, N)))
    seg0 = FieldSegment(SegmentId(0), phi[:half], ext0)
    seg1 = FieldSegment(SegmentId(1), phi[half:], ext1)
    field = Field("phi", (seg0, seg1), Placement({SegmentId(0): 0, SegmentId(1): 1}))
    assert field.covers(Extent.from_shape(phi.shape))


def test_gap_in_coverage_is_detected(phi: jnp.ndarray) -> None:
    # Row 3 is missing between the two segments.
    ext0 = Extent((slice(0, 3), slice(0, N), slice(0, N)))
    ext1 = Extent((slice(4, N), slice(0, N), slice(0, N)))
    seg0 = FieldSegment(SegmentId(0), phi[:3], ext0)
    seg1 = FieldSegment(SegmentId(1), phi[4:], ext1)
    field = Field("phi", (seg0, seg1), Placement({SegmentId(0): 0, SegmentId(1): 1}))
    assert not field.covers(Extent.from_shape(phi.shape))


def test_covers_checks_halo_expansion(phi: jnp.ndarray) -> None:
    # A segment that covers only the interior is not sufficient once the
    # 7-point stencil halo is added.
    interior = Extent((slice(1, N - 1), slice(1, N - 1), slice(1, N - 1)))
    seg = FieldSegment(SegmentId(0), phi[1 : N - 1, 1 : N - 1, 1 : N - 1], interior)
    field = Field("phi", (seg,), Placement({SegmentId(0): 0}))
    required = interior.expand(seven_point_laplacian.access_pattern)
    assert not field.covers(required)


# ---------------------------------------------------------------------------
# Single-process Field → Dispatch integration (degenerate case)
# ---------------------------------------------------------------------------


def test_single_process_field_dispatch_laplacian(phi: jnp.ndarray) -> None:
    """One Field, one segment, full domain — the degenerate single-rank case."""
    full = Extent.from_shape(phi.shape)
    seg = FieldSegment(SegmentId(0), phi, full)
    field = Field("phi", (seg,), Placement({SegmentId(0): 0}))

    interior = Extent((slice(1, N - 1), slice(1, N - 1), slice(1, N - 1)))
    required = interior.expand(seven_point_laplacian.access_pattern)
    assert field.covers(required)

    result = Dispatch(
        seven_point_laplacian,
        Region(interior),
        inputs=(seg.payload,),
    ).execute()
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
