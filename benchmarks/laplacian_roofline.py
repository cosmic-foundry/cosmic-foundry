"""CPU roofline benchmark for the 3-D seven-point Laplacian dispatch."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from typing import Any

import jax
import jax.numpy as jnp

from cosmic_foundry.kernels import Dispatch, Extent, Region, Stencil, op

FLOAT64_BYTES = 8
LAPLACIAN_BYTES_PER_CELL = 8 * FLOAT64_BYTES  # seven reads, one write
TRIAD_BYTES_PER_CELL = 3 * FLOAT64_BYTES  # two reads, one write


@op(
    access_pattern=Stencil.seven_point(),
    reads=("phi",),
    writes=("laplacian_phi",),
)
def seven_point_laplacian(phi: jax.Array, i: jax.Array, j: jax.Array, k: jax.Array):
    """Evaluate the 3-D nearest-neighbor Laplacian at indexed cells."""
    return (
        phi[i - 1, j, k]
        + phi[i + 1, j, k]
        + phi[i, j - 1, k]
        + phi[i, j + 1, k]
        + phi[i, j, k - 1]
        + phi[i, j, k + 1]
        - 6.0 * phi[i, j, k]
    )


@dataclass(frozen=True)
class RooflineResult:
    """Measured roofline summary for one benchmark run."""

    backend: str
    device: str
    n: int
    interior_cells: int
    repeats: int
    laplacian_seconds_best: float
    laplacian_effective_gb_s: float
    stream_triad_seconds_best: float
    stream_triad_gb_s: float
    memory_roofline_fraction: float


def make_phi(n: int) -> jax.Array:
    """Create a deterministic scalar field with nonzero curvature."""
    axes = jnp.indices((n, n, n), dtype=jnp.float64)
    return jnp.sin(0.07 * axes[0]) + jnp.cos(0.11 * axes[1]) + 0.001 * axes[2] ** 2


def run_laplacian(phi: jax.Array) -> jax.Array:
    """Run the public Dispatch path used by the benchmark."""
    n = int(phi.shape[0])
    extent = Extent((slice(1, n - 1), slice(1, n - 1), slice(1, n - 1)))
    return Dispatch(
        seven_point_laplacian,
        Region(extent),
        inputs=(phi,),
    ).execute()


def benchmark(n: int, repeats: int) -> RooflineResult:
    """Benchmark Laplacian throughput against a local STREAM-like triad."""
    phi = make_phi(n)
    laplacian = jax.jit(run_laplacian)
    triad = jax.jit(lambda a, b, scalar: a + scalar * b)

    laplacian(phi).block_until_ready()
    triad(phi, phi, 0.5).block_until_ready()

    laplacian_seconds = min(_time_call(lambda: laplacian(phi), repeats))
    triad_seconds = min(_time_call(lambda: triad(phi, phi, 0.5), repeats))

    interior_cells = (n - 2) ** 3
    laplacian_gb = interior_cells * LAPLACIAN_BYTES_PER_CELL / 1.0e9
    triad_gb = n**3 * TRIAD_BYTES_PER_CELL / 1.0e9
    laplacian_gb_s = laplacian_gb / laplacian_seconds
    triad_gb_s = triad_gb / triad_seconds

    return RooflineResult(
        backend=jax.default_backend(),
        device=str(jax.devices()[0]),
        n=n,
        interior_cells=interior_cells,
        repeats=repeats,
        laplacian_seconds_best=laplacian_seconds,
        laplacian_effective_gb_s=laplacian_gb_s,
        stream_triad_seconds_best=triad_seconds,
        stream_triad_gb_s=triad_gb_s,
        memory_roofline_fraction=laplacian_gb_s / triad_gb_s,
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
