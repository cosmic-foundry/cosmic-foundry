"""Diagnostic reductions and sinks."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import jax
import jax.numpy as jnp
import numpy as np

from cosmic_foundry.fields import Field
from cosmic_foundry.kernels import Extent, Region


class DiagnosticReducer(Protocol):
    """Structural protocol for one scalar diagnostic reduction."""

    name: str
    includes_boundary_flux: bool

    def reduce(
        self,
        fields: Mapping[str, Field],
        region: Region,
        rank: int,
        n_ranks: int,
    ) -> jax.Array:
        """Return a 0-d JAX array for this diagnostic."""


@dataclass(frozen=True)
class DiagnosticRecord:
    """One host-visible diagnostic row emitted at a named fence."""

    step: int
    time: float
    values: dict[str, float]


class DiagnosticSink(Protocol):
    """Host-visible destination for diagnostic records."""

    def write(self, record: DiagnosticRecord) -> None:
        """Write one diagnostic record."""


@dataclass(frozen=True)
class NullDiagnosticSink:
    """Diagnostic sink that discards records."""

    def write(self, record: DiagnosticRecord) -> None:
        """Discard *record*."""


@dataclass(frozen=True)
class TabSeparatedDiagnosticSink:
    """Append diagnostic records to a tab-separated text file."""

    path: str | Path
    columns: tuple[str, ...]

    def __init__(self, path: str | Path, columns: Sequence[str]) -> None:
        if not columns:
            msg = "DiagnosticSink requires at least one value column"
            raise ValueError(msg)
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "columns", tuple(columns))

    def write(self, record: DiagnosticRecord) -> None:
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


def collect_diagnostics(
    reducers: Sequence[DiagnosticReducer],
    fields: Mapping[str, Field],
    region: Region,
    *,
    step: int,
    time: float,
    rank: int,
    n_ranks: int,
) -> DiagnosticRecord:
    """Collect reducer outputs and materialize once at the diagnostic fence."""
    names = tuple(reducer.name for reducer in reducers)
    if len(set(names)) != len(names):
        msg = "DiagnosticReducer names must be unique"
        raise ValueError(msg)

    device_values = [
        jnp.asarray(r.reduce(fields, region, rank, n_ranks)) for r in reducers
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
            name: float(value) for name, value in zip(names, host_values, strict=False)
        },
    )


def global_sum(
    field: Field,
    region: Region,
    rank: int,
    *,
    axis_name: Hashable | None = None,
) -> jax.Array:
    """Sum field values over local interiors and optionally all-reduce them.

    Without *axis_name*, this returns the rank-local sum. Supplying a JAX
    parallel-map axis name applies ``jax.lax.psum`` and returns the global
    sum inside that mapped context.
    """
    local = jnp.asarray(0.0, dtype=jnp.float64)
    for segment in field.local_segments(rank):
        interior = segment.interior_extent or segment.extent
        overlap = _intersect_extents(interior, region.extent)
        if overlap is None:
            continue
        local = local + jnp.sum(
            segment.payload[_payload_slices(segment.extent, overlap)]
        )

    if axis_name is None:
        return local
    return jax.lax.psum(local, axis_name)


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
    "DiagnosticRecord",
    "DiagnosticReducer",
    "DiagnosticSink",
    "NullDiagnosticSink",
    "TabSeparatedDiagnosticSink",
    "collect_diagnostics",
    "global_sum",
]
