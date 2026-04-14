# Cosmic Foundry — Implementation Roadmap

This document is the high-level, time-ordered plan for building the
Cosmic Foundry engine. It is derived from RESEARCH.md — specifically
the union of capabilities in §6 and the engine-design implications in
§7 — and expresses a development sequence for a **fully self-contained
Python engine** whose performance is delivered by runtime code
generation rather than by a compiled C++ core.

The roadmap is intentionally strategic: each epoch is weeks-to-months
of real work, not a sprint. Granularity is "order of operations,"
not "task list." The expectation is that finer-grained plans will be
written per epoch as its entry criteria are met. Per-epoch files live
alongside this one and grow in detail as each epoch approaches.

---

## 1. Framing

**Goal.** A self-contained engine whose capability set is the union of
capabilities catalogued in RESEARCH.md §6 — covering uniform and
adaptive meshes, Newtonian through numerical-relativistic
(magneto)hydrodynamics, radiation transport, microphysics, gravity,
N-body, cosmology, particle / meshless / moving-mesh methods, and
stellar evolution.

**Principle — infrastructure first, physics second.** Every physics
module should land as a paper-scale effort on top of mature
primitives. Infrastructure mistakes in Epoch 1–2 cost an order of
magnitude more to fix than in any later epoch.

**Principle — easy case before hard case.** Within each subsystem,
ship the simplest instance before the general one: uniform grid
before AMR; Newtonian before special-relativistic before GR before
dynamical-spacetime; gray flux-limited diffusion before multigroup
M1; SPH before moving mesh.

**Principle — autodiff as a first-class tool.** JAX and SymPy
autodiff eliminate hand-derived Jacobians for stiff integrators,
primitive-variable recovery, BSSN/Z4c source terms, and parameter
sensitivity. Design physics modules so their Jacobians are obtained
from autodiff unless there is a demonstrated performance reason to
hand-code them.

**Principle — licensing discipline.** Capabilities with a
permissive-licensed reference code (RESEARCH.md §6.12) are studied
and adapted with attribution. Copyleft references are consulted
through published papers only; any reimplementation is clean-room.
Closed / collaboration-only codes (KEPLER, CHIMERA, FORNAX,
WhiskyTHC, private Arepo/GIZMO modules) contribute only through
their publications.

---

## 2. Technology baseline

These are initial commitments — revisitable, but changes should
require an ADR.

- **Python ≥3.11** as the single source language. No compiled
  extensions shipped from this repository; any native code is
  produced at runtime by a codegen backend.
- **JAX + XLA** as the primary array, autodiff, and JIT layer.
  `pjit` / `shard_map` provide device parallelism within a host,
  composed with `jax.distributed` (NCCL / GLOO, ADR-0003) for
  between-host collectives.
- **Secondary kernel backends**, wrapped behind a `@kernel`
  descriptor layer so they are interchangeable per-kernel. These
  backends are **designed for but not exercised in the early
  epochs** — only JAX is brought up first, with the others stubbed
  so that additional adapters can slot in later without surgery:
  - **Numba** — CPU SIMD and CUDA kernels where XLA's shape /
    control-flow constraints are limiting.
  - **Taichi** — particle / SPH / unstructured / moving-mesh
    workloads; mature simulation DSL.
  - **NVIDIA Warp** — GPU particle, tree, and neighbor kernels;
    Monte Carlo transport.
  - **Triton** — hand-tuned GPU kernels for the few cases that need
    full control.
- **`jax.distributed`** with NCCL (GPU) / GLOO (CPU) as the
  host-parallelism baseline (ADR-0003). `mpi4py` and `mpi4jax` are
  *not* baseline dependencies; they remain available as optional
  extras for sites where `jax.distributed` cannot initialize over
  the native interconnect.
- **h5py + parallel HDF5** for checkpoints and plotfiles. ADIOS2
  (Python) considered later.
- **NumPy + SciPy** for low-cost CPU work and reference
  implementations.
- **SymPy** for symbolic derivation of fluxes, Jacobians, and
  curvature tensors, with codegen into JAX or Numba.
- **pybind11 / ctypes** tolerated only as emergency escape hatches.
- **pytest + hypothesis** for testing; golden-file regression via
  HDF5 snapshot comparison.
- **Sphinx + MyST-NB** for documentation (already provisioned by
  miniforge).
- **PEP 621 `pyproject.toml`** with optional extras for each kernel
  backend, so users can install a minimum stack for their hardware.

---

## 3. Epochs

Each epoch is detailed in its own file. Stretch epochs are optional
and enter the schedule only once the baseline sequence is stable.

| # | File | Scope |
|--:|------|-------|
| 0 | [epoch-00-bootstrap.md](epoch-00-bootstrap.md) | Project scaffolding, packaging, CI, docs, ADR process, visualization scaffolding, `cosmic-foundry hello`. |
| 1 | [epoch-01-kernels.md](epoch-01-kernels.md) | Kernel descriptor, multi-backend adapters, `ShardedField`, parallel HDF5. |
| 2 | [epoch-02-mesh.md](epoch-02-mesh.md) | Uniform grid, block-structured AMR, task-graph driver, plotfile + Zarr writers. |
| 3 | [epoch-03-visualization.md](epoch-03-visualization.md) | Unit-aware plotting, in-engine JAX renderer, WebGPU viewer, visual-regression harness, public gallery. |
| 4 | [epoch-04-newtonian-hydro.md](epoch-04-newtonian-hydro.md) | Finite-volume Godunov, Riemann solvers, hydro test battery, first physics explainer. |
| 5 | [epoch-05-gravity-nbody.md](epoch-05-gravity-nbody.md) | Multigrid Poisson, particle infrastructure, tree gravity, FMM prototype. |
| 6 | [epoch-06-microphysics.md](epoch-06-microphysics.md) | EOS interface, reaction networks, cooling tables, opacities. |
| 7 | [epoch-07-mhd.md](epoch-07-mhd.md) | Ideal and non-ideal MHD, super-time-stepping, anisotropic transport. |
| 8 | [epoch-08-radiation.md](epoch-08-radiation.md) | Gray / multigroup FLD, two-moment M1, short characteristics. |
| 9 | [epoch-09-relativistic.md](epoch-09-relativistic.md) | SR, GR, GRMHD, dynamical-spacetime NR; BBH inspiral cinematic. |
| 10 | [epoch-10-particle-cosmology.md](epoch-10-particle-cosmology.md) | SPH / meshless methods, cosmology, halo finders, light-cones. |
| 11 | [epoch-11-moving-mesh.md](epoch-11-moving-mesh.md) | Stretch — Arepo-class moving Voronoi mesh. |
| 12 | [epoch-12-stellar-evolution.md](epoch-12-stellar-evolution.md) | Stretch — 1-D Lagrangian stellar structure with implicit solver. |
| 13 | [epoch-13-subgrid-observables.md](epoch-13-subgrid-observables.md) | Stretch — subgrid plugin interface, synthetic observables, in-situ rendering. |

---

## 4. Cross-cutting, continuous concerns

These grow every epoch; they are not tied to a single phase.

- **Testing.** Each physics epoch adds regression problems locked
  behind golden HDF5 outputs. Tests run across every supported
  kernel backend.
- **Performance.** Roofline analysis and strong / weak scaling
  tracked from Epoch 1 onward. Per-backend numbers are published so
  backend choice is empirical rather than ideological.
- **Documentation.** User guide, theory manual, and API reference
  advance alongside code. MyST-NB notebooks ship as part of docs.
- **Attribution.** Capabilities derived from a permissive-licensed
  reference cite that code and its paper at the algorithmic level.
  Copyleft references are cited but never copied.
- **Reproducibility.** Bitwise-reproducible runs are a goal for
  single-backend CPU execution (following GAMER-2, GADGET-4); full
  cross-backend bitwise reproducibility is not expected.
- **Visualization and communication.** Excellence at visualization
  is a core engine requirement, not a downstream concern. Every
  physics epoch from Epoch 4 onward ships a canonical live demo
  (interactive WebGPU explainer or notebook widget, perceptual
  colormaps, unit-labelled axes) alongside its regression
  benchmarks. The public gallery is regenerated per release with
  accessibility (WCAG 2.2 AA, colourblind-safe palettes, alt text)
  and mobile performance budgets (LCP < 2.5 s on a 4G profile,
  per-dataset bytes-on-wire targets) enforced in CI. Visual-
  regression references (figures, renders, short movies) live in
  Git LFS and are updated only by explicit tag-and-review, never
  auto-replaced. See ADR-0006 for the stack choices.

---

## 5. Crossroads / open decisions

These are flagged now and must be resolved before entering the
named epoch, via an ADR:

- ~~**Host-level parallelism model**~~ — resolved by ADR-0003:
  `jax.distributed` + NCCL / GLOO is the baseline between hosts,
  composed with `pjit` / `shard_map` within a host; MPI is an
  optional per-site fallback, not in the baseline.
- **In-engine renderer vs. yt hand-off** — whether the camera,
  slice sampler, and volume raymarcher live inside the engine
  (and are therefore shareable between CPU batch, GPU batch, and
  in-browser WebGPU) or are delegated to yt + widgyts. ADR-0006
  currently proposes in-engine; confirm before Epoch 3 starts.
- **Web streaming format — Zarr vs. ADIOS2** — the canonical
  output consumed by browser viewers. Expected to be Zarr v3 for
  its mature browser story with ADIOS2 retained for HPC analysis.
  Decide before Epoch 3.
- **Browser rendering target — WebGPU-first vs. WebGL2-first** —
  whether engine-side shaders are authored against WebGPU with a
  WebGL2 fallback, or vice versa. Decide before Epoch 3.
- **Scene-graph / camera ownership** — whether the engine ships
  its own minimal scene abstraction (transforms, cameras, lights)
  or adopts one from `three.js` / `pyvista`. Decide before
  Epoch 3.
- **Preferred particle / SPH backend** — Taichi vs NVIDIA Warp.
  Decide before Epoch 5.
- **Symbolic vs hand-written microphysics** — how much of the EOS
  and reaction-network machinery is SymPy-derived. Decide before
  Epoch 6.
- **Numerical-transcription discipline** — how to reconcile
  ADR-0005's ~100-LOC commit guideline with files like
  `aprox_rates.H` that are dense collections of analytic rate
  formulas, where the real defect defense is ADR-0007's
  golden-data verification, not diff-size reviewability.
  Problem framing and candidate approaches captured in
  [ADR-0008](../adr/ADR-0008-numerical-transcription-discipline.md)
  (stub). Decide before Epoch 6.
- **NR discretization** — whether dynamical-spacetime evolution in
  Epoch 9 stays on block AMR or adopts a DG / spectral
  discretization. Decide before Epoch 9.
- **Problem-setup surface** — pure Python API, YAML + validated
  schema, or a small domain-specific language. Decide before the
  first external user.
- **Compiled-extension exception** — when, if ever, to relax the
  "no compiled extensions" rule if a single kernel refuses to hit
  performance targets under any backend. Revisit whenever the
  question is forced by data.

---

## 6. Relationship to RESEARCH.md

Every epoch above traces back to specific capabilities in
RESEARCH.md §6:

| Epoch | RESEARCH.md §6 capabilities covered |
|------:|-------------------------------------|
| 0 | — (scaffolding; viz house style seeded from §6.11) |
| 1 | §6.9 (parallelism), §6.10 (I/O), §6.8 (solver infrastructure) |
| 2 | §6.1 (meshes and AMR), §6.10 (plotfiles + Zarr) |
| 3 | §6.11 (visualization and science communication), §6.10 (web streaming) |
| 4 | §6.2 (Newtonian hydro) |
| 5 | §6.4 (gravity and N-body) |
| 6 | §6.5 (EOS and microphysics), §6.3 (cooling) |
| 7 | §6.2 (MHD, non-ideal) |
| 8 | §6.3 (radiation transport) |
| 9 | §6.2 (SR / GR / NR), §6.8 (primitive recovery) |
| 10 | §6.1 (SPH / meshless), §6.4 (cosmology) |
| 11 | §6.1 (moving mesh) |
| 12 | §6.7 (stellar evolution) |
| 13 | §6.6 (subgrid), §6.10 (diagnostics) |

No epoch covers a capability that is not named in RESEARCH.md §6,
and every capability named in §6 appears in at least one epoch.
