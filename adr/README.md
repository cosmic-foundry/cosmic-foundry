# Architectural Decision Records

Each ADR captures one architectural decision: the context that
forced it, the choice made, the consequences, and the alternatives
considered. ADRs describe current architecture. When a decision
changes, edit the ADR in place — git history records what changed
and when. If a decision is entirely withdrawn, remove or archive the
ADR from the index. See
[ADR-0005 §Decision → ADR editing policy](ADR-0005-branch-pr-attribution-discipline.md#adr-editing-policy)
for the authoritative rule.

## How to use this index

- **On startup**, read this index. It is the canonical map of
  decisions in force.
- **When work touches a topic listed here**, read the full ADR
  before making changes. The index is a pointer, not a summary
  you can rely on for nuance.
- **When making a new architectural decision**, copy
  `adr-template.md` to `ADR-NNNN-<short-title>.md` and add a
  line to this index in the same PR.

## Index

ADRs are grouped by concern cluster to keep the family close to an
orthogonal basis (see `AI.md` → *Epoch retrospective* → ADR set as a
whole). The numbering reflects chronological order of adoption and is
preserved across regroupings.

### Engine foundation

Source language, kernel backend, host parallelism, and default
numerical precision — the load-bearing choices every later ADR builds on.

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
- [**ADR-0009**](ADR-0009-float64-default-precision.md) —
  Float64 is the default and only supported precision for all
  kernels and public APIs. `jax_enable_x64` is set at package
  import; no `dtype=` on public signatures yet; no ambient
  precision flag. Expected to be edited in place when
  mixed-precision experimentation begins, to add an explicit
  per-kernel opt-in for lower dtypes.

### Kernel model

The Op / Region / Policy / Dispatch vocabulary and the descriptors that
extend it (halo exchange, global reductions).

- [**ADR-0010**](ADR-0010-kernel-abstraction-model.md) —
  Kernel abstraction model: four named concepts (Op, Region, Policy,
  Dispatch) separating the computational, spatial, and execution axes.
  Op is a per-element callable with declared access pattern; Region is
  an iteration extent with optional batching; Policy is the execution
  organization (flat / tiled / warp-specialized); Dispatch is the
  dispatch unit composing one or more Ops with a Region and Policy.
  Only FlatPolicy is implemented in Epoch 1.
- [**ADR-0011**](ADR-0011-halo-fill-fence.md) —
  Halo fill fence: `HaloFillFence` descriptor + `HaloFillPolicy`
  execution split for ghost-cell exchange. The driver inserts fences
  before dispatches whose required footprint extends beyond the local
  segment interior; `HaloFillPolicy` executes the exchange via
  `jax.distributed`.
- [**ADR-0012**](ADR-0012-global-reduction-primitive.md) —
  Global reduction primitive: `DiagnosticReducer` protocol,
  `DiagnosticRecord` container, `DiagnosticSink` writer, and
  `global_sum` helper. Tab-separated `.diag` file per run; includes
  the boundary-flux balance test pattern for outflow BCs and a
  documentation requirement for conservation-law validity conditions.

### Process and verification

How work is organized — branch/PR discipline, the replication
workflow, transcription-heavy files, and the derivation lane for
physics capabilities.

- [**ADR-0005**](ADR-0005-branch-pr-attribution-discipline.md) —
  Branch, PR, commit-size, history, and attribution discipline for
  human and AI-agent contributors. Authoritative source; AI.md is
  an informal quick-reference kept aligned with this ADR.
- [**ADR-0007**](ADR-0007-replication-workflow.md) — Replication
  workflow: bounded-increment, verification-first. Every PR with
  a numerical claim ships with golden-data verification; spec
  docs and a per-target plan live under `replication/`; named
  exceptions for integration-time debugging are enumerated in
  `replication/README.md`.
- [**ADR-0008**](ADR-0008-numerical-transcription-discipline.md)
  *(stub)* — Numerical-transcription discipline for
  files like `aprox_rates.H` where the ~100-LOC guideline is
  ill-fitted and ADR-0007's verification-first discipline is
  what actually catches defects. Reserves the number and records
  the problem framing; final decision deferred to before Epoch 7.
- [**ADR-0013**](ADR-0013-derivation-first-lane.md) —
  Derivation-first lane for physics capabilities. Three named paths:
  Lane A (port-and-verify from a permissive reference), Lane B
  (clean-room from paper; required when the reference is
  copyleft-licensed), Lane C (first-principles origination for
  generalizations and novel work). Lanes B and C require a
  derivation document under `derivations/` with executable SymPy
  checks on load-bearing algebraic steps; Lean is available but
  not a required dependency. Operationalizes the licensing principle
  in `research/06-12-licensing.md` and `roadmap/index.md`.

### Organization and multi-repo architecture

How the project is structured across repositories — the platform/application
split, where observational data lives, and how application repos relate to
the platform and to each other.

- [**ADR-0014**](ADR-0014-platform-application-architecture.md) —
  Platform / application repository architecture: cosmic-foundry is the
  organizational platform providing computation infrastructure and manifest
  tooling; domain-specific application repos provide physics implementations
  and observational validation data. Introduces `cosmic_foundry.manifests`
  as the shared data-pipeline infrastructure.

### Documentation and visualization

Authoring and presentation of engine output — docs toolchain and
visualization stack.

- [**ADR-0004**](ADR-0004-sphinx-myst-docs-stack.md) —
  Sphinx + MyST-NB documentation stack with `sphinx-design`;
  `sphinx-autodoc2` for API reference; `sphinx-build -W` in CI.
  Hosted on GitHub Pages at
  `cosmic-foundry.github.io/cosmic-foundry/`. Interactivity is
  parameter-driven: sliders feed **live** simulation outputs computed
  in the browser by engine-authored WebGPU / WASM artifacts (per
  ADR-0006), not by a browser-side Python runtime. Rendered pages
  hide notebook-cell chrome by default. Theme (furo vs
  pydata-sphinx-theme) and CSS polish land with the first substantial
  docs PR.
- [**ADR-0006**](ADR-0006-visualization-stack.md) — Visualization
  and science-communication stack: Zarr v3 alongside HDF5;
  WebGPU-first browser target with WebGL2 fallback; perceptual
  colormaps via cmasher / cmocean; pyvista and vispy for desktop
  3-D; pytest-mpl plus an in-repo perceptual-diff harness for
  visual regressions.
