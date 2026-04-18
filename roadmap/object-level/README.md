# Object-Level Roadmap

This plane is the high-level, time-ordered plan for the platform and
simulation capabilities Cosmic Foundry will gain. It answers **what the
codebase is building**.

The companion [meta-level roadmap](../meta-level/README.md) answers how
those object-level claims are verified, validated, regenerated, and
audited.

## Framing

**Goal.** A platform whose computation capability set is the union of
capabilities cataloged in `RESEARCH.md` section 6 — covering uniform and
adaptive meshes, Newtonian through numerical-relativistic
(magneto)hydrodynamics, radiation transport, microphysics, gravity,
N-body, cosmology, particle / meshless / moving-mesh methods, and
stellar evolution — and whose platform services are reusable by every
downstream application repository.

**Platform and application roles.** Cosmic Foundry is the organizational
platform. Application repositories build on top of it. The platform is
intentionally heavy: general-purpose infrastructure lives here even when
a single application would not need all of it. Application repos are thin
on infrastructure and may be rich on physics.

**Principle - infrastructure first, physics second.** Every physics
module should land as a paper-scale effort on top of mature primitives.
Infrastructure mistakes in Epoch 1-2 cost an order of magnitude more to
fix than in any later epoch.

**Principle - platform before application.** Reusable platform
capabilities are delivered and stabilized before domain-specific physics
workflows are built on top of them in application repos.

**Principle - easy case before hard case.** Within each subsystem, ship
the simplest instance before the general one: uniform grid before AMR;
Newtonian before special-relativistic before GR; SPH before moving mesh.

**Principle - autodiff as a first-class tool.** JAX and SymPy autodiff
eliminate hand-derived Jacobians for stiff integrators,
primitive-variable recovery, BSSN/Z4c source terms, and parameter
sensitivity unless measured performance forces a hand-coded exception.

**Principle - licensing discipline.** Capabilities with a
permissive-licensed reference code are studied and adapted with
attribution. Copyleft references are consulted through published papers
only; any reimplementation is clean-room.

## Technology Baseline

These are initial commitments — revisitable, but changes should require
an ADR.

- Python >=3.11 as the single source language. No compiled extensions
  shipped from this repository; any native code is produced at runtime by
  a codegen backend.
- JAX + XLA as the primary array, autodiff, and JIT layer.
- `jax.distributed` with NCCL (GPU) / GLOO (CPU) as the host-parallelism
  baseline.
- h5py + parallel HDF5 for checkpoints and plotfiles.
- NumPy + SciPy for low-cost CPU work and reference implementations.
- SymPy for symbolic derivation of fluxes, Jacobians, and curvature
  tensors, with codegen into JAX or Numba.
- `requests` + `jsonschema` + `pyyaml` for manifest and specification
  infrastructure through the `[observational]` optional extra.
- pytest + hypothesis for testing; golden-file regression via HDF5
  snapshot comparison.
- Sphinx + MyST-NB for documentation.
- PEP 621 `pyproject.toml` with optional extras for each kernel backend
  and for manifest infrastructure.

## Epochs

Each object-level epoch is detailed in its own file. Stretch epochs are
optional and enter the schedule only once the baseline sequence is stable.

| # | File | Scope |
|--:|------|-------|
| 0 | [epoch-00-bootstrap.md](epoch-00-bootstrap.md) | Project scaffolding, packaging, CI, docs, ADR process, visualization scaffolding, `cosmic-foundry hello`. |
| 1 | [epoch-01-kernels.md](epoch-01-kernels.md) | Op / Region / Policy / Dispatch kernel interface, JAX `FlatPolicy`, Field placement, parallel HDF5. |
| 2 | [epoch-02-mesh.md](epoch-02-mesh.md) | Uniform grid, block-structured AMR, task-graph driver, plotfile + Zarr writers. |
| 3 | [epoch-03-platform-services.md](epoch-03-platform-services.md) | Manifest and specification infrastructure, provenance, comparison-result schema, problem-setup surface; application repository bootstrapping. |
| 4 | [epoch-04-visualization.md](epoch-04-visualization.md) | Unit-aware plotting, in-engine JAX renderer, WebGPU viewer, visual-regression harness, public gallery. |
| 5 | [epoch-05-newtonian-hydro.md](epoch-05-newtonian-hydro.md) | Finite-volume Godunov, Riemann solvers, hydro test battery, first physics explainer. |
| 6 | [epoch-06-gravity-nbody.md](epoch-06-gravity-nbody.md) | Multigrid Poisson, particle infrastructure, tree gravity, FMM prototype. |
| 7 | [epoch-07-microphysics.md](epoch-07-microphysics.md) | EOS interface, reaction networks, cooling tables, opacities. |
| 8 | [epoch-08-mhd.md](epoch-08-mhd.md) | Ideal and non-ideal MHD, super-time-stepping, anisotropic transport. |
| 9 | [epoch-09-radiation.md](epoch-09-radiation.md) | Gray / multigroup FLD, two-moment M1, short characteristics. |
| 10 | [epoch-10-relativistic.md](epoch-10-relativistic.md) | SR, GR, GRMHD, dynamical-spacetime NR; BBH inspiral cinematic. |
| 11 | [epoch-11-particle-cosmology.md](epoch-11-particle-cosmology.md) | SPH / meshless methods, cosmology, halo finders, light-cones. |
| 12 | [epoch-12-moving-mesh.md](epoch-12-moving-mesh.md) | Stretch — Arepo-class moving Voronoi mesh. |
| 13 | [epoch-13-stellar-evolution.md](epoch-13-stellar-evolution.md) | Stretch — 1-D Lagrangian solver infrastructure; stellar-physics application repo houses the physics application layer. |
| 14 | [epoch-14-subgrid-observables.md](epoch-14-subgrid-observables.md) | Stretch — subgrid plugin interface, synthetic observable hooks; application repos provide domain-specific implementations. |

## Crossroads / Open Decisions

These are object-level decisions that must be resolved before entering
the named epoch, via an ADR:

- Comparison-result schema — decide during Epoch 3.
- In-engine renderer vs. yt hand-off — decide before Epoch 4.
- Web streaming format — Zarr vs. ADIOS2 — decide before Epoch 4.
- Browser rendering target — WebGPU-first vs. WebGL2-first — decide
  before Epoch 4.
- Scene-graph / camera ownership — decide before Epoch 4.
- Preferred particle / SPH backend — Taichi vs NVIDIA Warp — decide
  before Epoch 6.
- Symbolic vs hand-written microphysics — decide before Epoch 7.
- NR discretization — decide before Epoch 10.
- Compiled-extension exception — revisit whenever forced by measured
  performance data.

## Relationship To RESEARCH.md

Every object-level physics epoch traces back to specific capabilities in
`RESEARCH.md` section 6. Epoch 3 is platform services rather than a
physics capability.

| Epoch | RESEARCH.md section 6 capabilities covered |
|------:|--------------------------------------------|
| 0 | scaffolding; visualization house style seeded from section 6.11 |
| 1 | section 6.9 parallelism, section 6.10 I/O, section 6.8 solver infrastructure |
| 2 | section 6.1 meshes and AMR, section 6.10 plotfiles + Zarr |
| 3 | platform services: manifest infrastructure, provenance, comparison-result schema |
| 4 | section 6.11 visualization and science communication, section 6.10 web streaming |
| 5 | section 6.2 Newtonian hydro |
| 6 | section 6.4 gravity and N-body |
| 7 | section 6.5 EOS and microphysics, section 6.3 cooling |
| 8 | section 6.2 MHD, non-ideal |
| 9 | section 6.3 radiation transport |
| 10 | section 6.2 SR / GR / NR, section 6.8 primitive recovery |
| 11 | section 6.1 SPH / meshless, section 6.4 cosmology |
| 12 | section 6.1 moving mesh |
| 13 | section 6.7 stellar evolution |
| 14 | section 6.6 subgrid, section 6.10 diagnostics |
