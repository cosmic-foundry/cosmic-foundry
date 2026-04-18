"""Tests for I/O and observability."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import jax.numpy as jnp
import numpy as np
import pytest

from cosmic_foundry.descriptor import AccessPattern, Extent, Region
from cosmic_foundry.io import HAS_PARALLEL_HDF5, merge_rank_files, write_array
from cosmic_foundry.map import Map, execute_pointwise
from cosmic_foundry.observability import StructuredFormatter, configure

# ---------------------------------------------------------------------------
# Shared Op
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


@pytest.fixture()
def laplacian_result(phi: jnp.ndarray) -> jnp.ndarray:
    region = Region(Extent((slice(1, N - 1), slice(1, N - 1), slice(1, N - 1))))
    return seven_point_laplacian.execute(phi, region=region)


# ---------------------------------------------------------------------------
# write_array
# ---------------------------------------------------------------------------


def test_write_array_round_trips_laplacian(
    tmp_path: Path, laplacian_result: jnp.ndarray
) -> None:
    dest = tmp_path / "laplacian.h5"
    write_array(dest, laplacian_result)

    with h5py.File(dest, "r") as f:
        stored = f["data"][()]

    assert stored.shape == laplacian_result.shape
    assert np.allclose(stored, 6.0)


def test_write_array_custom_dataset_name(
    tmp_path: Path, laplacian_result: jnp.ndarray
) -> None:
    dest = tmp_path / "out.h5"
    write_array(dest, laplacian_result, dataset="laplacian_phi")

    with h5py.File(dest, "r") as f:
        assert "laplacian_phi" in f
        assert np.allclose(f["laplacian_phi"][()], 6.0)


def test_write_array_accepts_numpy(tmp_path: Path) -> None:
    arr = np.ones((4, 4, 4))
    dest = tmp_path / "ones.h5"
    write_array(dest, arr)

    with h5py.File(dest, "r") as f:
        assert np.array_equal(f["data"][()], arr)


# ---------------------------------------------------------------------------
# merge_rank_files
# ---------------------------------------------------------------------------


def test_merge_rank_files_concatenates_along_axis0(tmp_path: Path) -> None:
    half = N // 2
    axes = jnp.indices((N, N, N), dtype=jnp.float64)
    phi = axes[0] ** 2 + axes[1] ** 2 + axes[2] ** 2

    # Simulate two ranks computing their interior half.
    interior = Extent((slice(1, N - 1), slice(1, N - 1), slice(1, N - 1)))
    full_result = seven_point_laplacian.execute(phi, region=Region(interior))

    rank0_result = full_result[:half]
    rank1_result = full_result[half:]

    path0 = tmp_path / "rank0.h5"
    path1 = tmp_path / "rank1.h5"
    write_array(path0, rank0_result)
    write_array(path1, rank1_result)

    merged_path = tmp_path / "merged.h5"
    merge_rank_files([path0, path1], merged_path)

    with h5py.File(merged_path, "r") as f:
        merged = f["data"][()]

    assert merged.shape == full_result.shape
    assert np.allclose(merged, np.asarray(full_result))


def test_merge_rank_files_rejects_empty_list(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one"):
        merge_rank_files([], tmp_path / "out.h5")


def test_has_parallel_hdf5_is_bool() -> None:
    assert isinstance(HAS_PARALLEL_HDF5, bool)


# ---------------------------------------------------------------------------
# Structured logging
# ---------------------------------------------------------------------------


def test_structured_formatter_produces_valid_json() -> None:
    record = logging.LogRecord(
        name="cosmic_foundry.kernels",
        level=logging.DEBUG,
        pathname="",
        lineno=0,
        msg="op.execute",
        args=(),
        exc_info=None,
    )
    record.__dict__["region_shape"] = [6, 6, 6]
    record.__dict__["n_blocks"] = 1

    fmt = StructuredFormatter()
    line = fmt.format(record)
    parsed = json.loads(line)

    assert parsed["level"] == "DEBUG"
    assert parsed["logger"] == "cosmic_foundry.kernels"
    assert parsed["event"] == "op.execute"
    assert parsed["region_shape"] == [6, 6, 6]
    assert parsed["n_blocks"] == 1


def test_write_array_emits_log_record(
    tmp_path: Path,
    laplacian_result: jnp.ndarray,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.DEBUG, logger="cosmic_foundry.io"):
        write_array(tmp_path / "out.h5", laplacian_result)

    events = [r.message for r in caplog.records]
    assert "io.write_array" in events


def test_merge_rank_files_emits_log_record(
    tmp_path: Path,
    laplacian_result: jnp.ndarray,
    caplog: pytest.LogCaptureFixture,
) -> None:
    p0, p1 = tmp_path / "r0.h5", tmp_path / "r1.h5"
    half = laplacian_result.shape[0] // 2
    write_array(p0, laplacian_result[:half])
    write_array(p1, laplacian_result[half:])

    with caplog.at_level(logging.DEBUG, logger="cosmic_foundry.io"):
        merge_rank_files([p0, p1], tmp_path / "merged.h5")

    events = [r.message for r in caplog.records]
    assert "io.merge_rank_files" in events


def test_configure_attaches_structured_handler() -> None:
    root = logging.getLogger()
    handler = logging.StreamHandler()
    configure(level=logging.DEBUG, handler=handler)
    assert isinstance(handler.formatter, StructuredFormatter)
    # Clean up: remove the handler so it does not leak into other tests.
    root.removeHandler(handler)
