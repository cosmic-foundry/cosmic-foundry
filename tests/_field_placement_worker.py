"""Subprocess worker for the multi-rank Field placement correctness harness.

Usage (called by the pytest multihost test; not run directly):
    python _field_placement_worker.py <rank> <num_processes> <coordinator_address>

Each worker:
 1. Initializes jax.distributed so the run is a genuine multi-process
    execution under JAX's gRPC coordination layer.
 2. Builds a local leaf DiscreteField covering its half of the domain
    (rows [0, half+1) for rank 0, rows [half-1, n) for rank 1) so the
    7-point stencil can evaluate at the seam without ghost exchange.
 3. Verifies local Field coverage, then dispatches the Laplacian.
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

        from cosmic_foundry.fields import (
            DiscreteField,
            Placement,
            SegmentId,
        )
        from cosmic_foundry.kernels import Dispatch, Extent, Region, Stencil, op

        n = 8
        half = n // 2  # 4

        # Build the full-domain field for slicing (cheap on CPU).
        axes = jnp.indices((n, n, n), dtype=jnp.float64)
        phi_full = axes[0] ** 2 + axes[1] ** 2 + axes[2] ** 2

        # Each rank gets its half plus a 1-cell halo at the seam so the
        # 7-point stencil is valid at local index 1 and local index half-1.
        # Both ranks share the same local shape (half+1, n, n) = (5, 8, 8).
        if rank == 0:
            local_phi = phi_full[: half + 1]  # global rows 0..4
        else:
            local_phi = phi_full[half - 1 :]  # global rows 3..7

        # Extents and region are expressed in local (0-based) coordinates.
        local_extent = Extent.from_shape(local_phi.shape)
        seg_id = SegmentId(rank)
        seg = DiscreteField(
            name="phi", segment_id=seg_id, payload=local_phi, extent=local_extent
        )
        placement = Placement({SegmentId(0): 0, SegmentId(1): 1})
        field = DiscreteField(name="phi", segments=(seg,), placement=placement)

        # Owned interior: local x in [1, half) avoids the boundary rows.
        owned_region = Region(
            Extent((slice(1, half), slice(1, n - 1), slice(1, n - 1)))
        )

        @op(
            access_pattern=Stencil.seven_point(),
            reads=("phi",),
            writes=("laplacian_phi",),
        )
        def laplacian(phi, i, j, k):  # type: ignore[no-untyped-def]
            return (
                phi[i - 1, j, k]
                + phi[i + 1, j, k]
                + phi[i, j - 1, k]
                + phi[i, j + 1, k]
                + phi[i, j, k - 1]
                + phi[i, j, k + 1]
                - 6.0 * phi[i, j, k]
            )

        # Verify coverage before dispatching.
        required = owned_region.extent.expand(laplacian.access_pattern)
        if not field.covers(required):
            raise RuntimeError("local segment does not cover owned region + halo")

        result = Dispatch(laplacian(local_phi), owned_region).execute()
        all_close = bool(jnp.allclose(result, 6.0))
        print(json.dumps({"rank": rank, "ok": True, "all_close_6": all_close}))

    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"rank": rank, "ok": False, "error": str(exc)}))


if __name__ == "__main__":
    main()
