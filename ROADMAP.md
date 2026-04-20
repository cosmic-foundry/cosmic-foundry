# Cosmic Foundry — Roadmap

For cross-cutting architectural decisions, see [`ARCHITECTURE.md`](ARCHITECTURE.md).
For near-term work and current status, see [`STATUS.md`](STATUS.md).

This file covers the long-horizon capability sequence. Items belong here when they
are not yet specified well enough to implement without further design discussion.
Once an item is fully specified and unblocked, it moves to `STATUS.md`.

---

## Planned continuous/ additions

**`DynamicManifold(PseudoRiemannianManifold)`**
— A manifold whose metric tensor is a dynamical field in the simulation state.
Required for full GR (3+1 ADM formalism). Planned alongside Epoch 12.

**`Connection` / `AffineConnection`**
— Covariant derivative; not a tensor field (inhomogeneous transformation law).
Required for curvature computations and parallel transport. Planned alongside
Epoch 12.

---

## Epoch sequence

The first two epochs establish the two foundational layers. Physics epochs then
add new continuous descriptions (new fields, new operators, new equations) that
the discrete and numerical layers evaluate.

### Foundation

| Epoch | Layer | Capability |
|-------|-------|------------|
| 0 | — | Project scaffolding: CI, pre-commit, documentation standards. ✓ |
| 1 | Continuous | `continuous/` ABCs: full manifold and field hierarchy, operators, boundary conditions, metric; coordinate structure (`Chart`, `Atlas`, `IdentityChart`, `SingleChartAtlas`); `SmoothManifold.atlas` constitutive. `foundation/` ABCs: `Set`, `Function`, `IndexedSet`, `IndexedFamily`. `discrete/` ABCs: `DiscreteField`, `DiscreteScalarField`, `DiscreteVectorField`. ✓ |
| 2 | Discrete | Cartesian grid as a concrete `IndexedSet` with coordinate geometry; cell and face structure. `DiscreteScalarField` and `DiscreteVectorField` backed by the grid. |
| 3 | Discrete | Discrete differential operators: stencil coefficients derived from continuous operators via SymPy; truncation error verified algebraically; formal operator composition on the grid. |
| 4 | Numerical | JAX evaluation layer: concrete field storage as `jax.Array`; JIT-compiled stencil application; explicit time integration; HDF5 I/O with provenance. |

### Physics

Each physics epoch adds new fields and equations to the continuous layer and
extends the discrete and numerical layers minimally to evaluate them.

| Epoch | Capability |
|-------|------------|
| 5 | Scalar transport: linear advection and diffusion on a Cartesian grid. First end-to-end simulation; validates the full pipeline. |
| 6 | Newtonian hydrodynamics: Euler equations, finite-volume Godunov, PPM reconstruction, HLLC/HLLE Riemann solvers. |
| 7 | Self-gravity: multigrid Poisson solver; particle infrastructure. |
| 8 | Microphysics: EOS interface, reaction networks, cooling tables, opacities. |
| 9 | MHD: ideal and resistive, constrained transport, super-time-stepping. |
| 10 | Radiation transport: gray FLD, multigroup FLD, two-moment M1. |
| 11 | AMR: adaptive mesh refinement hierarchy, coarse–fine interpolation, load balancing. |
| 12 | Special and general relativity: SR hydro, GR hydro/MHD on fixed spacetimes, dynamical spacetime via BSSN. |
| 13 | Particle cosmology: SPH, meshless methods, FRW integrator, halo finders. *(stretch)* |
| 14 | Moving mesh: Arepo-class Voronoi tessellation. *(stretch)* |
| 15 | Stellar evolution: 1-D Lagrangian solver with nuclear burning and mixing. *(stretch)* |
| 16 | Subgrid physics and synthetic observables: plugin interface, in-situ rendering. *(stretch)* |

---

## Platform & infrastructure

### Milestones

| Milestone | Capability |
|-----------|------------|
| M0 | Process discipline: branch/PR/commit/attribution standards. ✓ |
| M1 | Verification infrastructure: convergence testing helpers, formulas register, externally-grounded test pattern. ✓ |
| M2 | Documentation architecture: all live architectural decisions in `ARCHITECTURE.md`; `docs/` as API reference index. ✓ |
| M2.5 | Mathematical narrative documentation: executable MyST-NB notebooks that explain each layer of the hierarchy — what the formal concepts mean, how they relate, and why the code is structured the way it is. Notebooks run in CI, so every mathematical claim is machine-checked. Designed as a learning resource for contributors whose differential geometry background is thin; also exercises the test infrastructure in a direction orthogonal to unit tests. |
| M3 | Validation infrastructure: manifests, provenance sidecars, comparison-result schema. Planned alongside Epoch 4. |
| M4 | Reproducibility capsule tooling: self-executing builder. |
| M5 | Application-repo capsule integration and multi-repository evidence regeneration. |

### Per-epoch verification standard

Every physics epoch must satisfy this checklist before it is considered verified:

- Derivation document with SymPy checks for any new numerical scheme (Lanes B and C)
- At least one externally-grounded convergence test against an analytical solution
  or observational data (not an engine-generated golden file); where an analytical
  solution exists, the relevant `DiscreteField.approximates` is set so the check
  runs automatically
- Lane A/B/C classification stated in the PR description
