"""CPU roofline benchmark for a pointwise Function triad."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from typing import Any

import jax
import jax.numpy as jnp

from cosmic_foundry.computation.array import Array
from cosmic_foundry.computation.descriptor import Extent
from cosmic_foundry.computation.laplacian import seven_point_laplacian
from cosmic_foundry.computation.stencil import Stencil

FLOAT64_BYTES = 8
TRIAD_BYTES_PER_CELL = 3 * FLOAT64_BYTES  # two reads, one write


def _triad_fn(fields: tuple[Any, ...], i: Any, j: Any, k: Any) -> Any:
    a, b = fields[0], fields[1]
    return a[i, j, k] + 0.5 * b[i, j, k]


pointwise_triad = Stencil(fn=_triad_fn, radii=(0, 0, 0))


@dataclass(frozen=True)
class RooflineResult:
    """Measured roofline summary for one benchmark run."""

    backend: str
    device: str
    n: int
    cells: int
    repeats: int
    op_triad_seconds_best: float
    op_triad_seconds_median: float
    op_triad_gb_s: float
    stream_triad_seconds_best: float
    stream_triad_seconds_median: float
    stream_triad_gb_s: float
    memory_roofline_fraction: float


def make_phi(n: int) -> jax.Array:
    """Create a deterministic scalar field with nonzero curvature."""
    axes = jnp.indices((n, n, n), dtype=jnp.float64)
    return jnp.sin(0.07 * axes[0]) + jnp.cos(0.11 * axes[1]) + 0.001 * axes[2] ** 2


def run_laplacian(phi: jax.Array) -> jax.Array:
    """Run the Laplacian over the interior of *phi*."""
    n = int(phi.shape[0])
    extent = Extent((slice(1, n - 1), slice(1, n - 1), slice(1, n - 1)))
    return seven_point_laplacian.execute(Array((phi,)), extent=extent)


def run_op_triad(a: jax.Array, b: jax.Array) -> jax.Array:
    """Run a pointwise triad through the Stencil path."""
    n = int(a.shape[0])
    extent = Extent.from_shape((n, n, n))
    return pointwise_triad.execute(Array((a, b)), extent=extent)


def benchmark(n: int, repeats: int) -> RooflineResult:
    """Benchmark Op triad throughput against a direct JAX triad."""
    a = make_phi(n)
    b = make_phi(n) + 1.0
    op_triad = jax.jit(run_op_triad)
    triad = jax.jit(lambda a, b: a + 0.5 * b)

    op_triad(a, b).block_until_ready()
    triad(a, b).block_until_ready()

    op_timings = _time_call(lambda: op_triad(a, b), repeats)
    triad_timings = _time_call(lambda: triad(a, b), repeats)
    op_seconds_best = min(op_timings)
    op_seconds_median = statistics.median(op_timings)
    triad_seconds_best = min(triad_timings)
    triad_seconds_median = statistics.median(triad_timings)

    cells = n**3
    op_gb = cells * TRIAD_BYTES_PER_CELL / 1.0e9
    triad_gb = n**3 * TRIAD_BYTES_PER_CELL / 1.0e9
    op_gb_s = op_gb / op_seconds_median
    triad_gb_s = triad_gb / triad_seconds_median

    return RooflineResult(
        backend=jax.default_backend(),
        device=str(jax.devices()[0]),
        n=n,
        cells=cells,
        repeats=repeats,
        op_triad_seconds_best=op_seconds_best,
        op_triad_seconds_median=op_seconds_median,
        op_triad_gb_s=op_gb_s,
        stream_triad_seconds_best=triad_seconds_best,
        stream_triad_seconds_median=triad_seconds_median,
        stream_triad_gb_s=triad_gb_s,
        memory_roofline_fraction=op_gb_s / triad_gb_s,
    )


def _time_call(call: Any, repeats: int) -> list[float]:
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        call().block_until_ready()
        timings.append(time.perf_counter() - start)
    return timings


def main() -> None:
    """Run the benchmark and print a JSON result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=128, help="grid size per axis")
    parser.add_argument("--repeats", type=int, default=10, help="timed repetitions")
    args = parser.parse_args()

    if args.n < 4:
        parser.error("--n must be at least 4")
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    print(json.dumps(asdict(benchmark(args.n, args.repeats)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
