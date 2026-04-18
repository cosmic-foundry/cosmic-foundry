"""Diagnostic reductions and sinks."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np

from cosmic_foundry.descriptor import Extent, Region
from cosmic_foundry.map import Map
from cosmic_foundry.mesh import DistributedField
from cosmic_foundry.record import Record
from cosmic_foundry.sink import Sink


class DiagnosticReducer(Map):
    """Abstract base for one scalar diagnostic reduction.

    Map:
        domain   — ({f_h^i : Ω_h → ℝ}_i, Ω_h^int, rank, n_ranks) — named
                   discrete fields, the interior region, and rank metadata
        codomain — ℝ (a 0-d JAX array) — one scalar diagnostic value
        operator — execute({f_h^i}, region, rank, n_ranks) → scalar
    """

    name: str
    includes_boundary_flux: bool

    @abstractmethod
    def execute(
        self,
        fields: Mapping[str, DistributedField],
        region: Region,
        rank: int,
        n_ranks: int,
    ) -> jax.Array:
        """Return a 0-d JAX array for this diagnostic."""


@dataclass(frozen=True)
class DiagnosticRecord(Record):
    """One host-visible diagnostic row emitted at a named fence."""

    step: int
    time: float
    values: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        return {"step": self.step, "time": self.time, "values": dict(self.values)}


@dataclass(frozen=True)
class NullDiagnosticSink(Sink):
    """Diagnostic sink that discards records.

    Sink:
        domain — DiagnosticRecord (step, time, named scalar values)
        effect — none; record is silently discarded
    """

    def execute(self, record: DiagnosticRecord) -> None:
        """Discard *record*."""


@dataclass(frozen=True)
class TabSeparatedDiagnosticSink(Sink):
    """Append diagnostic records to a tab-separated text file.

    Sink:
        domain — DiagnosticRecord (step, time, named scalar values)
        effect — one tab-separated row appended to the file at path;
                 header row written on first call if the file is empty
    """

    path: str | Path
    columns: tuple[str, ...]

    def __init__(self, path: str | Path, columns: Sequence[str]) -> None:
        if not columns:
            msg = "TabSeparatedDiagnosticSink requires at least one value column"
            raise ValueError(msg)
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "columns", tuple(columns))

    def execute(self, record: DiagnosticRecord) -> None:
        """Append *record* using stable column order."""
        missing = [name for name in self.columns if name not in record.values]
        if missing:
            msg = f"DiagnosticRecord missing values for columns {missing!r}"
            raise ValueError(msg)

        path = Path(self.path)
        needs_header = not path.exists() or path.stat().st_size == 0
        with path.open("a", encoding="utf-8") as stream:
            if needs_header:
                stream.write("\t".join(("step", "time", *self.columns)) + "\n")
            values = [str(record.step), _format_float(record.time)]
            values.extend(_format_float(record.values[name]) for name in self.columns)
            stream.write("\t".join(values) + "\n")


@dataclass(frozen=True)
class CollectDiagnostics(Map):
    """Apply each reducer and materialise all scalars at the diagnostic fence.

    Map:
        domain   — ({f_h^i : Ω_h → ℝ}_i, [r_j]_j) — a named collection of
                   discrete fields and an ordered sequence of DiagnosticReducers
        codomain — (s_1, …, s_n) ∈ ℝⁿ — one scalar per reducer, host-visible
        operator — ({f_h^i}, [r_j]) ↦ (r_j.reduce({f_h^i}, region, rank, n_ranks))_j;
                   all device scalars are stacked into one jnp.array and
                   transferred to host with a single np.asarray call

    Exact: Θ = ∅ — the fence itself introduces no approximation; any
    approximation lives inside the individual DiagnosticReducer.reduce calls.
    """

    def execute(
        self,
        reducers: Sequence[DiagnosticReducer],
        fields: Mapping[str, DistributedField],
        region: Region,
        *,
        step: int,
        time: float,
        rank: int,
        n_ranks: int,
    ) -> DiagnosticRecord:
        names = tuple(reducer.name for reducer in reducers)
        if len(set(names)) != len(names):
            msg = "DiagnosticReducer names must be unique"
            raise ValueError(msg)

        device_values = [
            jnp.asarray(r.execute(fields, region, rank, n_ranks)) for r in reducers
        ]
        for name, value in zip(names, device_values, strict=False):
            if value.shape != ():
                msg = f"DiagnosticReducer {name!r} must return a scalar JAX array"
                raise ValueError(msg)

        host_values = (
            np.asarray(jnp.stack(device_values)) if device_values else np.array([])
        )
        return DiagnosticRecord(
            step=step,
            time=time,
            values={
                name: float(value)
                for name, value in zip(names, host_values, strict=False)
            },
        )


collect_diagnostics = CollectDiagnostics()


@dataclass(frozen=True)
class GlobalSum(Map):
    """Sum field values over local interiors and optionally all-reduce them.

    Map:
        domain   — f_h : Ω_h^int → ℝ, a discrete scalar field on interior
                   grid points, intersected with the given region
        codomain — ℝ (a real number; field evaluated at a single point)
        operator — (f_h, region) ↦ Σ_{x ∈ Ω_h^int ∩ region} f_h(x)

    Unweighted grid-point sum. To approximate ∫_Ω f dΩ, multiply by h^d
    where h is the grid spacing and d is the spatial dimension.

    Without *axis_name*, returns the rank-local sum. Supplying a JAX
    parallel-map axis name applies ``jax.lax.psum`` and returns the global
    sum inside that mapped context.

    Exact: Θ = ∅ — unweighted sum; no approximation introduced.
    """

    def execute(
        self,
        field: DistributedField,
        region: Region,
        rank: int,
        *,
        axis_name: Hashable | None = None,
    ) -> jax.Array:
        local = jnp.asarray(0.0, dtype=jnp.float64)
        for segment in field.local_segments(rank):
            extent = segment.extent
            payload = segment.payload
            interior: Extent = segment.interior_extent or extent
            overlap = _intersect_extents(interior, region.extent)
            if overlap is None:
                continue
            local = local + jnp.sum(payload[_payload_slices(extent, overlap)])

        if axis_name is None:
            return local
        return cast(jax.Array, jax.lax.psum(local, axis_name))


global_sum = GlobalSum()


def _format_float(value: float) -> str:
    return f"{value:.17g}"


def _payload_slices(parent: Extent, child: Extent) -> tuple[slice, ...]:
    return tuple(
        slice(
            child_slice.start - parent_slice.start,
            child_slice.stop - parent_slice.start,
        )
        for parent_slice, child_slice in zip(parent.slices, child.slices, strict=False)
    )


def _intersect_extents(a: Extent, b: Extent) -> Extent | None:
    if a.ndim != b.ndim:
        msg = "Cannot intersect Extents with different ndim"
        raise ValueError(msg)
    slices: list[slice] = []
    for sa, sb in zip(a.slices, b.slices, strict=False):
        start = max(sa.start, sb.start)
        stop = min(sa.stop, sb.stop)
        if start >= stop:
            return None
        slices.append(slice(start, stop))
    return Extent(tuple(slices))


__all__ = [
    "CollectDiagnostics",
    "DiagnosticRecord",
    "DiagnosticReducer",
    "GlobalSum",
    "NullDiagnosticSink",
    "TabSeparatedDiagnosticSink",
    "collect_diagnostics",
    "global_sum",
]
