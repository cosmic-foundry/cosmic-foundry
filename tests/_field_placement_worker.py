"""Subprocess worker for the multi-rank Field placement correctness harness."""

from __future__ import annotations

import json
import sys


def main() -> None:
    rank = int(sys.argv[1])
    num_processes = int(sys.argv[2])
    coordinator_address = sys.argv[3]

    try:
        import jax

        jax.config.update("jax_enable_x64", True)
        jax.distributed.initialize(
            coordinator_address=coordinator_address,
            num_processes=num_processes,
            process_id=rank,
            initialization_timeout=30,
        )

        import jax.numpy as jnp

        from cosmic_foundry.computation.array import Array
        from cosmic_foundry.computation.descriptor import Extent
        from cosmic_foundry.computation.stencil import Stencil

        n = 8
        half = n // 2

        axes = jnp.indices((n, n, n), dtype=jnp.float64)
        phi_full = axes[0] ** 2 + axes[1] ** 2 + axes[2] ** 2

        if rank == 0:
            local_phi = phi_full[: half + 1]
        else:
            local_phi = phi_full[half - 1 :]

        def _fn(fields, i, j, k):
            phi = fields[0]
            return (
                phi[i - 1, j, k]
                + phi[i + 1, j, k]
                + phi[i, j - 1, k]
                + phi[i, j + 1, k]
                + phi[i, j, k - 1]
                + phi[i, j, k + 1]
                - 6.0 * phi[i, j, k]
            )

        laplacian = Stencil(fn=_fn, radii=(1, 1, 1))
        owned_extent = Extent((slice(1, half), slice(1, n - 1), slice(1, n - 1)))
        result = laplacian.execute(Array((local_phi,)), extent=owned_extent)
        all_close = bool(jnp.allclose(result, 6.0))
        print(json.dumps({"rank": rank, "ok": True, "all_close_6": all_close}))

    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"rank": rank, "ok": False, "error": str(exc)}))


if __name__ == "__main__":
    main()
