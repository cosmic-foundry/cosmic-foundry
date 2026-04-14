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
written per epoch as its entry criteria are met.

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
permissive-licensed reference code (RESEARCH.md §6.11) are studied
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
  `pjit` / `shard_map` provide device- and host-parallelism on top
  of MPI.
- **Secondary kernel backends**, wrapped behind a `@kernel`
  descriptor layer so they are interchangeable per-kernel:
  - **Numba** — CPU SIMD and CUDA kernels where XLA's shape /
    control-flow constraints are limiting.
  - **Taichi** — particle / SPH / unstructured / moving-mesh
    workloads; mature simulation DSL.
  - **NVIDIA Warp** — GPU particle, tree, and neighbor kernels;
    Monte Carlo transport.
  - **Triton** — hand-tuned GPU kernels for the few cases that need
    full control.
- **mpi4py** for message passing; `mpi4jax` and `jax.distributed` for
  JAX interop.
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

### Epoch 0 — Project bootstrap

Establish the scaffolding that every later epoch assumes:

- Python package layout: `cosmic_foundry/`, `tests/`, `docs/`,
  `examples/`, `benchmarks/`.
- `pyproject.toml` with backend extras.
- Pre-commit hooks: black, ruff, mypy.
- GitHub Actions CI: lint, type-check, build, unit tests on CPU.
  Manual-dispatch GPU job scaffolded for when runners exist.
- Coding standards document and an ADR directory seeded with the
  decisions in §2.
- A first `cosmic-foundry hello` entrypoint that initializes MPI,
  prints rank/size, and exits cleanly under every kernel backend.

**Exit criterion:** CI is green on main; a developer can clone,
install, and run `pytest` and `cosmic-foundry hello` in under ten
minutes on a fresh machine that has miniforge.

### Epoch 1 — Kernel abstraction and multi-backend core

The one piece of infrastructure we genuinely have to invent:

- A `Kernel` descriptor — a declarative spec of a stencil or
  element-wise operation over typed arrays with named axes — plus
  adapters that lower it to JAX, Numba, Taichi, Warp, or Triton at
  runtime.
- A `ShardedField` distributed-array primitive built on
  `jax.distributed` + mpi4py for cross-device / cross-host cases.
- Parallel HDF5 I/O via h5py+MPI.
- Deterministic structured logging and error handling.

**Exit criterion:** a 3-D 7-point Laplacian benchmark runs under
every backend at documented roofline fractions, and a multi-rank
correctness test proves backend and sharding equivalence.

### Epoch 2 — Mesh and AMR

Bring up the mesh hierarchy the physics modules will live on:

- Uniform structured grid with ghost-cell exchange and domain
  decomposition.
- Block-structured AMR hierarchy expressed as JAX pytrees of blocks
  with cell / face / edge / node centering, subcycling in time, and
  refinement-flux correction.
- Task-graph driver for asynchronous dependency scheduling across
  ranks and devices.
- Plotfile writer and yt-compatible metadata.

**Exit criterion:** a second-order advection test converges at
design order on AMR, runs identically on CPU and GPU, and produces
rank-invariant output.

### Epoch 3 — Newtonian hydrodynamics

The first physics module and the template for every subsequent one:

- Finite-volume Godunov with PPM reconstruction (SymPy-derived
  stencils).
- Riemann solvers: HLLC, HLLE, Roe.
- CFL time-stepping and passive scalars.
- Golden regression suite: Sod, Sedov, Noh, blast wave,
  Kelvin–Helmholtz, Rayleigh–Taylor.

**Exit criterion:** the standard hydro test battery matches reference
solutions across all kernel backends.

### Epoch 4 — Self-gravity and N-body

Close the loop with gravitational dynamics:

- Geometric multigrid Poisson on the AMR hierarchy (JAX where
  possible; Taichi fallback for irregular smoothers).
- Particle infrastructure — cell-in-cloud deposition, neighbor
  search — on Warp or Taichi.
- Barnes–Hut tree gravity with Ewald summation for periodic BCs.
- FMM prototype.

**Exit criterion:** Zel'dovich pancake and hydrostatic-equilibrium
tests match reference solutions.

### Epoch 5 — Magnetohydrodynamics and non-ideal transport

Extend the hydro module into the MHD regime:

- Constrained-transport ideal MHD — flux-CT and staggered vector-
  potential formulations.
- Super-time-stepping for parabolic terms.
- Resistive, Hall, and ambipolar MHD.
- Anisotropic thermal conduction and viscosity.

**Exit criterion:** Orszag–Tang, MHD rotor, and MHD blast wave match
literature benchmarks.

### Epoch 6 — Microphysics sub-layer

Bring up the equations of state and reaction networks that later
physics modules depend on:

- Abstract EOS interface with ideal-gas, Helmholtz, piecewise
  polytropic, and tabulated nuclear finite-T implementations.
  Tables are JAX-jittable piecewise interpolants.
- Reaction-network engine with autodiff-generated Jacobians,
  α-network reference, and a path to large adaptive networks.
- Primordial and metal cooling tables.
- Radiation opacities.

**Exit criterion:** thermonuclear flame and primordial cooling
benchmarks match published results.

### Epoch 7 — Radiation transport

Close the radiation-hydrodynamics loop:

- Gray flux-limited diffusion.
- Multigroup FLD.
- Two-moment M1 with reduced speed of light.
- Short-characteristics variant.
- Interfaces left open for DG and Monte Carlo (the latter is a
  natural fit for Warp).

**Exit criterion:** shadow, linear-wave, and radiating-shock tests
pass.

### Epoch 8 — Relativistic physics

The hardest grid-based physics target:

- Special-relativistic hydrodynamics.
- GR hydro and GRMHD on fixed stationary spacetimes, with robust
  primitive-variable recovery (multiple schemes, autodiff
  Jacobians).
- Dynamical-spacetime evolution via BSSN or Z4c with moving-puncture
  gauge. SymPy derives the Einstein source terms.

**Exit criterion:** Fishbone–Moncrief torus and binary-black-hole
inspiral benchmarks.

### Epoch 9 — Particle / meshless hydrodynamics and cosmology

The second mesh paradigm and the cosmological stack:

- SPH with modern pressure–energy and density-independent variants;
  neighbor loops on Taichi or Warp.
- Meshless finite-mass and finite-volume methods.
- Comoving FRW integrator.
- 2LPT initial conditions.
- On-the-fly FOF and SUBFIND-style halo finders.
- Light-cones and high-dynamic-range power-spectrum estimator.

**Exit criterion:** a cosmological box reproduces a published
reference at fixed resolution within documented tolerance.

### Stretch Epoch 10 — Moving Voronoi mesh

The Arepo-class replication target. Deferred until the above are
stable so the engine can absorb the complexity. Likely implemented
on Taichi or Warp with SciPy-based CPU fallbacks for correctness
tests.

### Stretch Epoch 11 — Stellar evolution

1-D Lagrangian stellar structure with an implicit solver (JAX
autodiff for the block-tridiagonal Jacobian), adaptive mesh and
timestep, coupled nuclear burning, rotation, and mixing; a binary
evolution interface compatible with the multi-D explosive modules
so that progenitor states flow naturally into merger and supernova
runs.

### Stretch Epoch 12 — Subgrid physics and observables

- Plugin interface for cooling / star formation / stellar and SN
  feedback / AGN seeding and feedback / chemical enrichment, so that
  EAGLE-, COLIBRE-, and FIRE-class recipes can be expressed inside
  the engine.
- On-the-fly synthetic EUV, X-ray, and spectral-line observables.
- In-situ visualization hooks.

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

---

## 5. Crossroads / open decisions

These are flagged now and must be resolved before entering the
named epoch, via an ADR:

- **Host-level parallelism model** — JAX `pjit` / `shard_map` vs
  explicit mpi4py, or both. Likely both, with the former inside a
  node and the latter between nodes. Decide before Epoch 1 stable.
- **Preferred particle / SPH backend** — Taichi vs NVIDIA Warp.
  Decide before Epoch 4.
- **Symbolic vs hand-written microphysics** — how much of the EOS
  and reaction-network machinery is SymPy-derived. Decide before
  Epoch 6.
- **NR discretization** — whether dynamical-spacetime evolution in
  Epoch 8 stays on block AMR or adopts a DG / spectral
  discretization. Decide before Epoch 8.
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
| 0 | — (scaffolding) |
| 1 | §6.9 (parallelism), §6.10 (I/O), §6.8 (solver infrastructure) |
| 2 | §6.1 (meshes and AMR), §6.10 (plotfiles) |
| 3 | §6.2 (Newtonian hydro) |
| 4 | §6.4 (gravity and N-body) |
| 5 | §6.2 (MHD, non-ideal) |
| 6 | §6.5 (EOS and microphysics), §6.3 (cooling) |
| 7 | §6.3 (radiation transport) |
| 8 | §6.2 (SR / GR / NR), §6.8 (primitive recovery) |
| 9 | §6.1 (SPH / meshless), §6.4 (cosmology) |
| 10 | §6.1 (moving mesh) |
| 11 | §6.7 (stellar evolution) |
| 12 | §6.6 (subgrid), §6.10 (diagnostics) |

No epoch covers a capability that is not named in RESEARCH.md §6,
and every capability named in §6 appears in at least one epoch.
