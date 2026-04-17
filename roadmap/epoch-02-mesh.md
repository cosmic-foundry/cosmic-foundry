# Epoch 2 — Mesh and AMR

> Part of the [Cosmic Foundry roadmap](index.md).

## Design prerequisites

Two design decisions must be made **before** the task-graph driver is
implemented — they cannot be resolved mid-epoch without restructuring
work already in flight.

### Field name → Dispatch input binding

Epoch 1 `Op` metadata declares `reads=("phi",)` and `writes=("laplacian_phi",)`,
but `Dispatch.inputs` accepts positional raw arrays with no enforcement that
input[0] corresponds to the "phi" field.  The Epoch 2 task-graph driver must
answer: *given that a Dispatch reads field "phi", where in the field registry
is phi, which local segment belongs to this rank, and which input position does
it bind to?*

Design the binding layer first.  Options include a `fields: dict[str, Field]`
argument on `Dispatch`, a separate `FieldBinding` object, or a driver-owned
field-resolution step that extracts local payloads and constructs the
positional inputs before calling `execute`.  Record the decision as an
amendment to ADR-0010 before opening the task-graph PR.

### Halo fill fence

`Field.covers(required)` validates that coverage *exists*, but there is no
mechanism for the driver to *fill* missing halos.  The Epoch 1 multi-rank
worker sidestepped this by pre-slicing overlapping arrays from a locally
available full domain; production domain decomposition cannot do that.

Before implementing ghost-cell exchange, define the fence primitive: something
like `HaloFillFence(field, region, access_pattern)` that the driver inserts
before any `Dispatch` whose required footprint extends beyond a segment's
interior.  This is where `jax.distributed` collectives live.  The
`Field`/`Placement`/`Region` metadata model has all the information needed;
the orchestration layer does not yet exist.

---

## What this epoch delivers

Bring up the mesh hierarchy the physics modules will live on:

- Uniform structured grid with ghost-cell exchange and domain
  decomposition.
- Block-structured AMR hierarchy expressed as JAX pytrees of blocks
  with cell / face / edge / node centering, subcycling in time, and
  refinement-flux correction.
- Task-graph driver for asynchronous dependency scheduling across
  ranks and devices.
- Plotfile writer and yt-compatible metadata.
- Zarr v3 writer with an OME-Zarr-style multiscale pyramid
  convention emitted alongside the plotfile, so every AMR output
  is simultaneously HPC-analysis-ready (HDF5 / plotfile) and
  browser-streamable (Zarr). Pinned by ADR-0006.

**Exit criterion:** a second-order advection test converges at
design order on AMR, runs identically on CPU and GPU, produces
rank-invariant output, and writes a matching Zarr pyramid that
the Epoch 3 viewer MVP consumes without hand editing.
