"""Subprocess worker for the multi-rank Field placement correctness harness.

Usage (called by the pytest multihost test; not run directly):
    python _field_placement_worker.py <rank> <num_processes> <coordinator_address>

Each worker:
 1. Initializes jax.distributed so the run is a genuine multi-process
    execution under JAX's gRPC coordination layer.
 2. Builds a local array covering its half of the domain
    (rows [0, half+1) for rank 0, rows [half-1, n) for rank 1) so the
    7-point stencil can evaluate at the seam without ghost exchange.
 3. Dispatches the Laplacian on the owned interior.
 4. Prints one line of JSON:
      {"rank": <int>, "ok": true,  "all_close_6": <bool>}
   or {"rank": <int>, "ok": false, "error": "<msg>"}
"""

from __future__ import annotations

import json
import sys


def main() -> None:
    rank = int(sys.argv[1])
    num_processes = int(sys.argv[2])
    coordinator_address = sys.argv[3]

    try:
        # jax_enable_x64 must be set before any computation; distributed
        # init must happen before the first JAX array operation.
        import jax

        jax.config.update("jax_enable_x64", True)
        jax.distributed.initialize(
            coordinator_address=coordinator_address,
            num_processes=num_processes,
            process_id=rank,
            initialization_timeout=30,
        )

        import jax.numpy as jnp

        from cosmic_foundry.computation.descriptor import Extent
        from cosmic_foundry.computation.stencil import execute_pointwise
        from cosmic_foundry.theory.function import Function

        n = 8
        half = n // 2  # 4

        # Build the full-domain field for slicing (cheap on CPU).
        axes = jnp.indices((n, n, n), dtype=jnp.float64)
        phi_full = axes[0] ** 2 + axes[1] ** 2 + axes[2] ** 2

        # Each rank gets its half plus a 1-cell halo at the seam so the
        # 7-point stencil is valid at local index 1 and local index half-1.
        if rank == 0:
            local_phi = phi_full[: half + 1]  # global rows 0..4
        else:
            local_phi = phi_full[half - 1 :]  # global rows 3..7

        owned_extent = Extent((slice(1, half), slice(1, n - 1), slice(1, n - 1)))

        from dataclasses import dataclass
        from typing import Any

        @dataclass(frozen=True)
        class Laplacian(Function):
            @property
            def radii(self) -> tuple[int, ...]:
                return (1, 1, 1)

            def execute(self, phi: Any, *, extent: Extent) -> Any:
                return execute_pointwise(self, extent, phi)

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

        laplacian = Laplacian()

        result = laplacian.execute(local_phi, extent=owned_extent)
        all_close = bool(jnp.allclose(result, 6.0))
        print(json.dumps({"rank": rank, "ok": True, "all_close_6": all_close}))

    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"rank": rank, "ok": False, "error": str(exc)}))


if __name__ == "__main__":
    main()
