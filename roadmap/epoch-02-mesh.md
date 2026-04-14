# Epoch 2 — Mesh and AMR

> Part of the [Cosmic Foundry roadmap](index.md).

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
