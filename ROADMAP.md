# Cosmic Foundry — Roadmap

For cross-cutting architectural decisions, see [`ARCHITECTURE.md`](ARCHITECTURE.md).
For near-term work and current status, see [`STATUS.md`](STATUS.md).

This file covers the long-horizon capability sequence. Items belong here when they
are not yet specified well enough to implement without further design discussion.
Once an item is fully specified and unblocked, it moves to `STATUS.md`.

---

## Architecture

The codebase is organized into four packages with a strict dependency order:

**Foundation (`foundation/`)** — primitive mathematical abstractions shared by
all layers: `Set`, `Function`, `IndexedSet`, `IndexedFamily`. No floats.

**Continuous (`continuous/`)** — the problem in its true mathematical form.
Manifolds, smooth fields, differential operators, boundary conditions. Everything
is infinite-dimensional and coordinate-free. No arrays, no grids, no floats.

**Discrete (`discrete/`)** — scheme description on finite index sets. A
`DiscreteField` is a `Function[IndexedSet, V]`; it inherits from `foundation/`
vertically (is-a) and optionally references `continuous/` horizontally (has-a)
via the `approximates` property. When `approximates` is set, the discrete object
declares itself a finite approximation of the named continuous object, enabling
automatic convergence checks at computation time. When `approximates` is `None`,
the discrete object is a primary mathematical object — the data IS the object,
with no continuous antecedent (e.g. a field loaded from a MESA progenitor).
Stencil coefficients are exact rationals. This layer is still symbolic: it
describes the discretization without evaluating it.

`foundation/`, `continuous/`, and `discrete/` are **symbolic-reasoning layers**.
Their shared identity: they describe mathematical structure symbolically, without
numerical evaluation. They may import from stdlib, `cosmic_foundry`, or approved
symbolic-reasoning packages (`sympy`). JAX, NumPy, and other numerical packages
are excluded by identity, not by a blanket third-party ban. Adding a package to
the approved list requires justification against the symbolic-reasoning identity.

**Numerical (`computation/`)** — JAX evaluates the discrete description. Cell
values become `jax.Array`; stencil application becomes a JIT-compiled kernel;
time integration becomes a scan. This is the only layer that touches floats.

The continuous-to-discrete transition is the design of a numerical scheme. The
discrete-to-numerical transition is an evaluation step. Keeping them separate
means a scheme can be inspected, verified, and swapped without touching the
evaluator.

---

## Open architectural questions

**Is scheme choice a first-class concept?**
A finite-difference discretization of ∇² is a precise mathematical act: choose
a grid, choose an approximation order, derive stencil coefficients. The
`approximates` property on `DiscreteField` establishes the has-a link between
a discrete object and its continuous counterpart, but does not make scheme choice
(e.g. "second-order centered finite difference of the Laplacian") a first-class
object. An open question is whether a formal `Discretization` — a callable that
maps a `DifferentialOperator` + grid + order to a discrete stencil — belongs in
`discrete/`, or whether scheme choice remains implicit in how discrete objects
are constructed.

**What is the formal PDE object in the continuous layer?**
Conservation laws like ∂ρ/∂t + ∇·(ρv) = 0 are statements about continuous
fields. Before discretizing, we may want to express them as formal objects in
`continuous/`. The right interface is unclear and may only become clear once we
have a working discretization to invert from.

**What do SymPy-backed continuous objects look like?**
The symbolic-reasoning identity makes SymPy available in `continuous/` and
`discrete/`. The natural use is analytical field representations — a concrete
`ScalarField` backed by a SymPy expression `f(x, y) = sin(πx)sin(πy)` — which
would make `approximates` algebraically live: stencil derivation and truncation
error analysis could be done in code rather than in documentation. The interface
for SymPy-backed fields (evaluatable analytical forms, coordinate handling) is
not yet designed.

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
| 1 | Continuous | `continuous/` ABCs: full manifold and field hierarchy, operators, boundary conditions, metric. `foundation/` ABCs: `Set`, `Function`, `IndexedSet`, `IndexedFamily`. `discrete/` ABCs: `DiscreteField`, `DiscreteScalarField`, `DiscreteVectorField`. ✓ |
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
| M3 | Validation infrastructure: manifests, provenance sidecars, comparison-result schema. Planned alongside Epoch 4. |
| M4 | Reproducibility capsule tooling: self-executing builder. |
| M5 | Application-repo capsule integration and multi-repository evidence regeneration. |

### Per-epoch verification standard

Every physics epoch must satisfy this checklist before it is considered verified:

- Derivation document with SymPy checks for any new numerical scheme (Lanes B and C)
- Entry in the formulas register (`replication/formulas.md`) for each physics formula
- At least one externally-grounded convergence test against an analytical solution
  or observational data (not an engine-generated golden file); where an analytical
  solution exists, the relevant `DiscreteField.approximates` is set so the check
  runs automatically
- Lane A/B/C classification stated in the PR description
