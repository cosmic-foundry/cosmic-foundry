# Architectural Decision Records

Each ADR captures one architectural decision: the context that
forced it, the choice made, the consequences, and the
alternatives considered. ADRs are append-only — a decision is
revised by superseding it with a new ADR, not by editing the
old one.

## How to use this index

- **On startup**, read this index. It is the canonical map of
  decisions in force.
- **When work touches a topic listed here**, read the full ADR
  before making changes. The index is a pointer, not a summary
  you can rely on for nuance.
- **When making a new architectural decision**, copy
  `adr-template.md` to `ADR-NNNN-<short-title>.md`, mark it
  Proposed, and add a line to this index in the same PR.

## Index

- **ADR-0001 — ADR-0005:** Reserved for the Epoch-0 seed ADRs
  enumerated in `roadmap/epoch-00-bootstrap.md` §0.6. Not yet
  written.
- [**ADR-0006**](ADR-0006-visualization-stack.md) — Visualization
  and science-communication stack: Zarr v3 alongside HDF5;
  WebGPU-first browser target with WebGL2 fallback; perceptual
  colormaps via cmasher / cmocean; pyvista and vispy for desktop
  3-D; pytest-mpl plus an in-repo perceptual-diff harness for
  visual regressions.
- [**ADR-0007**](ADR-0007-replication-workflow.md) — Replication
  workflow: bounded-increment, verification-first. Every PR with
  a numerical claim ships with golden-data verification; spec
  docs and a per-target plan live under `replication/`; named
  exceptions for integration-time debugging are enumerated in
  `replication/README.md`.
