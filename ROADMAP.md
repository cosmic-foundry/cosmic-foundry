# Cosmic Foundry — Roadmap

For cross-cutting architectural decisions, see [`ARCHITECTURE.md`](ARCHITECTURE.md).
For near-term work and current status, see [`STATUS.md`](STATUS.md).

This file covers the long-horizon capability sequence. Items belong here when they
are not yet specified well enough to implement without further design discussion.
Once an item is fully specified and unblocked, it moves to `STATUS.md`.

---

## Architecture

The codebase is organized around three layers, each with a distinct role:

**Continuous (`theory/`)** — the problem in its true mathematical form.
Manifolds, smooth fields, differential operators, boundary conditions. Everything
is infinite-dimensional and coordinate-free. No arrays, no grids, no floats.

**Discrete** — a chosen discretization of the continuous problem. A grid is a
concrete `Region` with a finite cell count and exact spacing. Stencil coefficients
are exact rationals derived from the continuous operator. Discrete fields are
indexed families of values, one per cell. This layer is still symbolic: it
describes the discretization without evaluating it.

**Numerical (`computation/`)** — JAX evaluates the discrete description. Cell
values become `jax.Array`; stencil application becomes a JIT-compiled kernel;
time integration becomes a scan. This is the only layer that touches floats.

The continuous-to-discrete transition is the design of a numerical scheme. The
discrete-to-numerical transition is an evaluation step. Keeping them separate
means a scheme can be inspected, verified, and swapped without touching the
evaluator.

---

## Open architectural questions

**What does the continuous-to-discrete transition look like as code?**
A finite-difference discretization of ∇² is a precise mathematical act: choose
a grid, choose an approximation order, derive stencil coefficients. Should that
be a formal object (a `Discretization` that maps a `DifferentialOperator` to a
discrete stencil), or is the discrete layer just built directly without a formal
bridge? This determines whether scheme choice is a first-class concept.

**Does a discrete field know its continuous counterpart?**
A cell-centered density array approximates a `ScalarField`. Should the discrete
field carry a reference to the continuous field it approximates — establishing a
formal approximation relationship — or are they separate objects that share an ABC?

**What is the formal PDE object in the continuous layer?**
Conservation laws like ∂ρ/∂t + ∇·(ρv) = 0 are statements about continuous
fields. Before discretizing, we may want to express them as formal objects in
`theory/`. The right interface is unclear and may only become clear once we have
a working discretization to invert from.

---

## Planned theory additions

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
| 1 | Continuous | `theory/` ABCs: full manifold and field hierarchy, operators, boundary conditions, metric. ✓ |
| 2 | Discrete | Cartesian grid as a concrete `Region`; coordinate geometry; cell and face structure. Discrete scalar and vector fields indexed by the grid. |
| 3 | Discrete | Discrete differential operators: stencil coefficients derived from continuous operators; formal operator composition on the grid. |
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
| M3 | Validation infrastructure: manifests, provenance sidecars, comparison-result schema. Planned alongside Epoch 4. |
| M4 | Reproducibility capsule tooling: self-executing builder. |
| M5 | Application-repo capsule integration and multi-repository evidence regeneration. |

### Per-epoch verification standard

Every physics epoch must satisfy this checklist before it is considered verified:

- Derivation document with SymPy checks for any new numerical scheme (Lanes B and C)
- Entry in the formulas register (`replication/formulas.md`) for each physics formula
- At least one externally-grounded convergence test against an analytical solution
  or observational data (not an engine-generated golden file)
- Lane A/B/C classification stated in the PR description
