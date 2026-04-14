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

- [**ADR-0001**](ADR-0001-python-with-runtime-codegen.md) —
  Python-only engine with runtime code generation: Python ≥3.11 as
  the single source language; no compiled extensions shipped from
  this repository; native code produced at runtime by codegen
  backends.
- [**ADR-0002**](ADR-0002-jax-primary-kernel-backend.md) — JAX + XLA
  as the primary kernel backend. Numba, Taichi, NVIDIA Warp, and
  Triton accommodated as optional extras behind a `@kernel`
  descriptor layer but not exercised in Epoch 0–1.
- [**ADR-0003**](ADR-0003-jax-distributed-host-parallelism.md) —
  `jax.distributed` + NCCL (GPU) / GLOO (CPU) as the host-parallelism
  baseline. Single-layer programming model with `pjit` / `shard_map`
  within a host; MPI is available as an optional per-site fallback
  but not in the baseline dependencies.
- [**ADR-0004**](ADR-0004-sphinx-myst-docs-stack.md) *(Proposed)* —
  Sphinx + MyST-NB documentation stack with `sphinx-design`;
  `sphinx-autodoc2` for API reference; `sphinx-build -W` in CI.
  Interactivity is parameter-driven (sliders feeding pre-computed
  or JIT'd simulation outputs via the WebGPU viewer and
  `holoviews` + `bokeh`), not browser-side-Python code execution.
  Rendered pages hide notebook-cell chrome by default. Theme
  (furo vs pydata-sphinx-theme), CSS polish, and hosting
  (RTD / Pages) deferred to follow-up ADRs alongside the first
  substantial docs PR.
- [**ADR-0005**](ADR-0005-branch-pr-attribution-discipline.md) —
  Branch, PR, commit-size, history, and attribution discipline for
  human and AI-agent contributors. Authoritative source; AI.md is
  an informal quick-reference kept aligned with this ADR.
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
