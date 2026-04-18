# Object-Level Architecture

This plane records architecture for the engine, platform services, and
application-facing capability surface. It answers **what Cosmic Foundry
is building**.

Meta-level architecture lives separately in
[meta-level/README.md](../meta-level/README.md) and answers how
object-level claims are verified, validated, regenerated, and audited.

## Engine Foundation

Source language, kernel backend, host parallelism, and default numerical
precision — the load-bearing choices every later object-level ADR builds
on.

- [**ADR-0001**](ADR-0001-python-with-runtime-codegen.md) —
  Python-only engine with runtime code generation: Python >=3.11 as the
  single source language; no compiled extensions shipped from this
  repository; native code produced at runtime by codegen backends.
- [**ADR-0002**](ADR-0002-jax-primary-kernel-backend.md) — JAX + XLA as
  the primary kernel backend. Numba, Taichi, NVIDIA Warp, and Triton
  accommodated as optional extras behind a `@kernel` descriptor layer but
  not exercised in Epoch 0-1.
- [**ADR-0003**](ADR-0003-jax-distributed-host-parallelism.md) —
  `jax.distributed` + NCCL (GPU) / GLOO (CPU) as the host-parallelism
  baseline. Single-layer programming model with `pjit` / `shard_map`
  within a host; MPI is available as an optional per-site fallback but
  not in the baseline dependencies.
- [**ADR-0009**](ADR-0009-float64-default-precision.md) — Float64 is the
  default and only supported precision for all kernels and public APIs.
  `jax_enable_x64` is set at package import; no `dtype=` on public
  signatures yet; no ambient precision flag. Expected to be edited in
  place when mixed-precision experimentation begins, to add an explicit
  per-kernel opt-in for lower dtypes.

## Field And Function Formalism

The organizing concepts: fields as functions on manifolds, and functions
as relationships between mathematical objects.

- [**ADR-0016**](ADR-0016-field-map-formalism.md) — Field / function formalism:
  `Field(Function)` hierarchy with `ScalarField`, `TensorField`;
  `ContinuousField` (Θ = ∅) and `PatchFunction` (Θ = {h}) as concrete
  scalar fields; every operator class is a function with a documented
  domain, codomain, and operator; desiderata are not function parameters.

## Kernel Model

The Op / Region / Policy / Dispatch vocabulary and the descriptors that
extend it.

- [**ADR-0010**](ADR-0010-kernel-abstraction-model.md) — Kernel
  abstraction model: four named concepts (Op, Region, Policy, Dispatch)
  separating the computational, spatial, and execution axes. Only
  FlatPolicy is implemented in Epoch 1.
- [**ADR-0011**](ADR-0011-halo-fill-fence.md) — Halo fill fence:
  `HaloFillFence` descriptor + `HaloFillPolicy` execution split for
  ghost-cell exchange. The driver inserts fences before dispatches whose
  required footprint extends beyond the local segment interior.
- [**ADR-0012**](ADR-0012-global-reduction-primitive.md) — Global
  reduction primitive: `DiagnosticReducer` protocol, `DiagnosticRecord`
  container, `DiagnosticSink` writer, and `global_sum` helper.

## Organization And Platform Services

How the repository family is structured and which infrastructure belongs
in the platform.

- [**ADR-0014**](ADR-0014-platform-application-architecture.md) —
  Platform / application repository architecture: cosmic-foundry is the
  organizational platform providing computation infrastructure and
  manifest tooling; domain-specific application repos provide physics
  implementations and observational validation data. This ADR bridges to
  the meta-level plane because manifest infrastructure carries validation
  and provenance contracts.

## Documentation And Visualization

Authoring and presentation of engine output.

- [**ADR-0004**](ADR-0004-sphinx-myst-docs-stack.md) — Sphinx + MyST-NB
  documentation stack with `sphinx-design`, `sphinx-autodoc2` for API
  reference, and `sphinx-build -W` in CI.
- [**ADR-0006**](ADR-0006-visualization-stack.md) — Visualization and
  science-communication stack: Zarr v3 alongside HDF5, WebGPU-first
  browser target with WebGL2 fallback, perceptual colormaps, desktop 3-D
  tools, and visual-regression infrastructure.
