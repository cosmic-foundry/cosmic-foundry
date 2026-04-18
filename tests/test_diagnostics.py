"""Tests for diagnostic reductions and sinks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from cosmic_foundry.diagnostics import (
    DiagnosticRecord,
    NullDiagnosticSink,
    TabSeparatedDiagnosticSink,
    collect_diagnostics,
    global_sum,
)
from cosmic_foundry.fields import (
    Field,
    FieldSegment,
    Placement,
    SegmentId,
    allocate_field,
)
from cosmic_foundry.kernels import Extent, Region, Stencil
from cosmic_foundry.mesh import UniformGrid


@dataclass(frozen=True)
class SumReducer:
    name: str
    field_name: str
    includes_boundary_flux: bool = False

    def reduce(
        self,
        fields: dict[str, Field],
        region: Region,
        rank: int,
        n_ranks: int,
    ) -> jax.Array:
        return global_sum(fields[self.field_name], region, rank)


@dataclass(frozen=True)
class VectorReducer:
    name: str = "bad_vector"
    includes_boundary_flux: bool = False

    def reduce(
        self,
        fields: dict[str, Field],
        region: Region,
        rank: int,
        n_ranks: int,
    ) -> jax.Array:
        return jnp.array([1.0, 2.0])


def _grid_1d() -> UniformGrid:
    return UniformGrid.create(
        domain_origin=(0.0,),
        domain_size=(1.0,),
        n_cells=(8,),
        blocks_per_axis=(2,),
        n_ranks=1,
    )


def _field_with_payloads(values: tuple[jax.Array, ...]) -> Field:
    field = allocate_field("rho", _grid_1d(), Stencil((1,)))
    segments = tuple(
        FieldSegment(
            segment_id=segment.segment_id,
            payload=value,
            extent=segment.extent,
            interior_extent=segment.interior_extent,
        )
        for segment, value in zip(field.segments, values, strict=True)
    )
    return Field(field.name, segments, field.placement)


def test_global_sum_returns_jax_scalar_without_host_materialization() -> None:
    segment = FieldSegment(
        SegmentId(0),
        jnp.arange(6.0, dtype=jnp.float64),
        Extent((slice(0, 6),)),
    )
    field = Field("rho", (segment,), Placement({SegmentId(0): 0}))

    result = global_sum(field, Region(Extent((slice(1, 5),))), rank=0)

    assert result.shape == ()
    assert result == pytest.approx(10.0)


def test_global_sum_uses_interiors_and_does_not_count_halos() -> None:
    left = jnp.array([-1000.0, 1.0, 2.0, 3.0, 4.0, -1000.0])
    right = jnp.array([-2000.0, 5.0, 6.0, 7.0, 8.0, -2000.0])
    field = _field_with_payloads((left, right))

    result = global_sum(field, Region(Extent((slice(0, 8),))), rank=0)

    assert result == pytest.approx(36.0)


def test_global_sum_restricts_to_region() -> None:
    left = jnp.array([-1000.0, 1.0, 2.0, 3.0, 4.0, -1000.0])
    right = jnp.array([-2000.0, 5.0, 6.0, 7.0, 8.0, -2000.0])
    field = _field_with_payloads((left, right))

    result = global_sum(field, Region(Extent((slice(2, 6),))), rank=0)

    assert result == pytest.approx(18.0)


def test_collect_diagnostics_materializes_one_record() -> None:
    segment = FieldSegment(
        SegmentId(0),
        jnp.arange(4.0, dtype=jnp.float64),
        Extent((slice(0, 4),)),
    )
    field = Field("rho", (segment,), Placement({SegmentId(0): 0}))

    record = collect_diagnostics(
        (SumReducer("total_mass", "rho"),),
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
    with pytest.raises(ValueError, match="unique"):
        collect_diagnostics(
            (SumReducer("total", "rho"), SumReducer("total", "rho")),
            {},
            Region(Extent((slice(0, 1),))),
            step=0,
            time=0.0,
            rank=0,
            n_ranks=1,
        )


def test_collect_diagnostics_requires_scalar_outputs() -> None:
    with pytest.raises(ValueError, match="scalar"):
        collect_diagnostics(
            (VectorReducer(),),
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

    sink.write(DiagnosticRecord(0, 0.0, {"total_mass": 1.25, "cfl_max": 0.4}))
    sink.write(DiagnosticRecord(1, 0.5, {"total_mass": 1.5, "cfl_max": 0.45}))

    assert path.read_text(encoding="utf-8").splitlines() == [
        "step\ttime\ttotal_mass\tcfl_max",
        "0\t0\t1.25\t0.40000000000000002",
        "1\t0.5\t1.5\t0.45000000000000001",
    ]


def test_tab_separated_sink_rejects_missing_column(tmp_path: Path) -> None:
    sink = TabSeparatedDiagnosticSink(tmp_path / "run.diag", ("total_mass",))

    with pytest.raises(ValueError, match="missing"):
        sink.write(DiagnosticRecord(0, 0.0, {}))


def test_null_sink_discards_records() -> None:
    NullDiagnosticSink().write(DiagnosticRecord(0, 0.0, {"total_mass": 1.0}))
