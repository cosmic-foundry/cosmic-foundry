"""Tests for diagnostic reductions and sinks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from cosmic_foundry.descriptor import Extent, Region
from cosmic_foundry.diagnostics import (
    DiagnosticRecord,
    DiagnosticReducer,
    NullDiagnosticSink,
    TabSeparatedDiagnosticSink,
    collect_diagnostics,
    global_sum,
)
from cosmic_foundry.field import DiscreteField
from cosmic_foundry.mesh import Block, partition_domain
from cosmic_foundry.record import Array, ComponentId, Placement


def _make_1d_mesh(
    n_cells: int = 8, n_blocks: int = 2, n_ranks: int = 1
) -> Array[Block]:
    return partition_domain(
        domain_origin=(0.0,),
        domain_size=(float(n_cells),),
        n_cells=(n_cells,),
        blocks_per_axis=(n_blocks,),
        n_ranks=n_ranks,
    )


@dataclass(frozen=True)
class SumReducer(DiagnosticReducer):
    name: str
    field_name: str
    includes_boundary_flux: bool = False

    def execute(
        self,
        mesh: Array[Block],
        fields: dict[str, Array[DiscreteField]],
        region: Region,
        rank: int,
        n_ranks: int,
    ) -> jax.Array:
        return global_sum(mesh, fields[self.field_name], region, rank)


@dataclass(frozen=True)
class VectorReducer(DiagnosticReducer):
    name: str = "bad_vector"
    includes_boundary_flux: bool = False

    def execute(
        self,
        mesh: Array[Block],
        fields: dict[str, Array[DiscreteField]],
        region: Region,
        rank: int,
        n_ranks: int,
    ) -> jax.Array:
        return jnp.array([1.0, 2.0])


def test_global_sum_returns_jax_scalar() -> None:
    mesh = _make_1d_mesh(n_cells=6, n_blocks=1)
    field = Array(
        elements=(
            DiscreteField(name="rho", payload=jnp.arange(6.0, dtype=jnp.float64)),
        ),
        placement=Placement({ComponentId(0): 0}),
    )

    result = global_sum(mesh, field, Region(Extent((slice(1, 5),))), rank=0)

    assert result.shape == ()
    assert result == pytest.approx(10.0)


def test_global_sum_sums_over_interior_only() -> None:
    """GlobalSum over both blocks sums interior values, not halos."""
    mesh = _make_1d_mesh(n_cells=8, n_blocks=2)
    # Block 0 interior: [0, 4), block 1 interior: [4, 8)
    field = Array(
        elements=(
            DiscreteField(name="rho", payload=jnp.array([1.0, 2.0, 3.0, 4.0])),
            DiscreteField(name="rho", payload=jnp.array([5.0, 6.0, 7.0, 8.0])),
        ),
        placement=Placement({ComponentId(0): 0, ComponentId(1): 0}),
    )

    result = global_sum(mesh, field, Region(Extent((slice(0, 8),))), rank=0)

    assert result == pytest.approx(36.0)


def test_global_sum_restricts_to_region() -> None:
    mesh = _make_1d_mesh(n_cells=8, n_blocks=2)
    field = Array(
        elements=(
            DiscreteField(name="rho", payload=jnp.array([1.0, 2.0, 3.0, 4.0])),
            DiscreteField(name="rho", payload=jnp.array([5.0, 6.0, 7.0, 8.0])),
        ),
        placement=Placement({ComponentId(0): 0, ComponentId(1): 0}),
    )

    # Region [2, 6) overlaps block 0 at [2,4) and block 1 at [4,6)
    # Block 0 payload[2:4] = [3, 4]; block 1 payload[0:2] = [5, 6]
    result = global_sum(mesh, field, Region(Extent((slice(2, 6),))), rank=0)

    assert result == pytest.approx(18.0)


def test_collect_diagnostics_materializes_one_record() -> None:
    mesh = _make_1d_mesh(n_cells=4, n_blocks=1)
    field = Array(
        elements=(
            DiscreteField(name="rho", payload=jnp.arange(4.0, dtype=jnp.float64)),
        ),
        placement=Placement({ComponentId(0): 0}),
    )

    record = collect_diagnostics(
        (SumReducer("total_mass", "rho"),),
        mesh,
        {"rho": field},
        Region(Extent((slice(0, 4),))),
        step=7,
        time=0.125,
        rank=0,
        n_ranks=1,
    )

    assert record == DiagnosticRecord(
        step=7,
        time=0.125,
        values={"total_mass": 6.0},
    )


def test_collect_diagnostics_rejects_duplicate_names() -> None:
    mesh = _make_1d_mesh(n_cells=4, n_blocks=1)
    with pytest.raises(ValueError, match="unique"):
        collect_diagnostics(
            (SumReducer("total", "rho"), SumReducer("total", "rho")),
            mesh,
            {},
            Region(Extent((slice(0, 1),))),
            step=0,
            time=0.0,
            rank=0,
            n_ranks=1,
        )


def test_collect_diagnostics_requires_scalar_outputs() -> None:
    mesh = _make_1d_mesh(n_cells=4, n_blocks=1)
    with pytest.raises(ValueError, match="scalar"):
        collect_diagnostics(
            (VectorReducer(),),
            mesh,
            {},
            Region(Extent((slice(0, 1),))),
            step=0,
            time=0.0,
            rank=0,
            n_ranks=1,
        )


def test_tab_separated_sink_writes_header_and_rows(tmp_path: Path) -> None:
    path = tmp_path / "run.diag"
    sink = TabSeparatedDiagnosticSink(path, ("total_mass", "cfl_max"))

    sink.execute(DiagnosticRecord(0, 0.0, {"total_mass": 1.25, "cfl_max": 0.4}))
    sink.execute(DiagnosticRecord(1, 0.5, {"total_mass": 1.5, "cfl_max": 0.45}))

    assert path.read_text(encoding="utf-8").splitlines() == [
        "step\ttime\ttotal_mass\tcfl_max",
        "0\t0\t1.25\t0.40000000000000002",
        "1\t0.5\t1.5\t0.45000000000000001",
    ]


def test_tab_separated_sink_rejects_missing_column(tmp_path: Path) -> None:
    sink = TabSeparatedDiagnosticSink(tmp_path / "run.diag", ("total_mass",))

    with pytest.raises(ValueError, match="missing"):
        sink.execute(DiagnosticRecord(0, 0.0, {}))


def test_null_sink_discards_records() -> None:
    NullDiagnosticSink().execute(DiagnosticRecord(0, 0.0, {"total_mass": 1.0}))
