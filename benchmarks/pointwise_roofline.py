"""CPU roofline benchmark for a pointwise Dispatch triad."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from typing import Any, ClassVar

import jax
import jax.numpy as jnp

from cosmic_foundry.kernels import AccessPattern, Dispatch, Extent, Op, Region, Stencil

FLOAT64_BYTES = 8
TRIAD_BYTES_PER_CELL = 3 * FLOAT64_BYTES  # two reads, one write


@dataclass(frozen=True)
class SevenPointLaplacian(Op):
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

    reads: ClassVar[tuple[str, ...]] = ("phi",)
    writes: ClassVar[tuple[str, ...]] = ("laplacian_phi",)

    @property
    def access_pattern(self) -> AccessPattern:
        return Stencil.seven_point()

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


@dataclass(frozen=True)
class RooflineResult:
    """Measured roofline summary for one benchmark run."""

    backend: str
    device: str
    n: int
    cells: int
    repeats: int
    dispatch_triad_seconds_best: float
    dispatch_triad_seconds_median: float
    dispatch_triad_gb_s: float
    stream_triad_seconds_best: float
    stream_triad_seconds_median: float
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
        seven_point_laplacian(phi),
        Region(extent),
    ).execute()


@dataclass(frozen=True)
class PointwiseTriad(Op):
    """STREAM-like pointwise triad: c = a + 0.5 * b.

    Map:
        domain   — (a, b): two DiscreteFields on Ω_h
        codomain — c: DiscreteField on Ω_h
        operator — c_{ijk} = a_{ijk} + 0.5 * b_{ijk}

    Exact: Θ = ∅ — pointwise arithmetic; no approximation.
    """

    reads: ClassVar[tuple[str, ...]] = ("a", "b")
    writes: ClassVar[tuple[str, ...]] = ("c",)

    @property
    def access_pattern(self) -> AccessPattern:
        return Stencil((0, 0, 0))

    def _fn(self, a: Any, b: Any, i: Any, j: Any, k: Any) -> Any:
        return a[i, j, k] + 0.5 * b[i, j, k]


pointwise_triad = PointwiseTriad()


def run_dispatch_triad(a: jax.Array, b: jax.Array) -> jax.Array:
    """Run a pointwise triad through the public Dispatch path."""
    n = int(a.shape[0])
    extent = Extent.from_shape((n, n, n))
    return Dispatch(
        pointwise_triad(a, b),
        Region(extent),
    ).execute()


def benchmark(n: int, repeats: int) -> RooflineResult:
    """Benchmark Dispatch triad throughput against a direct JAX triad."""
    a = make_phi(n)
    b = make_phi(n) + 1.0
    dispatch_triad = jax.jit(run_dispatch_triad)
    triad = jax.jit(lambda a, b: a + 0.5 * b)

    dispatch_triad(a, b).block_until_ready()
    triad(a, b).block_until_ready()

    dispatch_timings = _time_call(lambda: dispatch_triad(a, b), repeats)
    triad_timings = _time_call(lambda: triad(a, b), repeats)
    dispatch_seconds_best = min(dispatch_timings)
    dispatch_seconds_median = statistics.median(dispatch_timings)
    triad_seconds_best = min(triad_timings)
    triad_seconds_median = statistics.median(triad_timings)

    cells = n**3
    dispatch_gb = cells * TRIAD_BYTES_PER_CELL / 1.0e9
    triad_gb = n**3 * TRIAD_BYTES_PER_CELL / 1.0e9
    dispatch_gb_s = dispatch_gb / dispatch_seconds_median
    triad_gb_s = triad_gb / triad_seconds_median

    return RooflineResult(
        backend=jax.default_backend(),
        device=str(jax.devices()[0]),
        n=n,
        cells=cells,
        repeats=repeats,
        dispatch_triad_seconds_best=dispatch_seconds_best,
        dispatch_triad_seconds_median=dispatch_seconds_median,
        dispatch_triad_gb_s=dispatch_gb_s,
        stream_triad_seconds_best=triad_seconds_best,
        stream_triad_seconds_median=triad_seconds_median,
        stream_triad_gb_s=triad_gb_s,
        memory_roofline_fraction=dispatch_gb_s / triad_gb_s,
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
