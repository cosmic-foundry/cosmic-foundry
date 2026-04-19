# Cosmic Foundry — Roadmap

For cross-cutting architectural decisions, see [`ARCHITECTURE.md`](ARCHITECTURE.md).
For the current codebase state and planned modules, see [`STATUS.md`](STATUS.md).

---

## Simulation capabilities

| Epoch | Capability |
|-------|------------|
| 0 | Project scaffolding: CI, pre-commit, documentation standards, entry point. ✓ |
| 1 | Kernel abstraction, field placement, HDF5 I/O, deterministic logging. ✓ |
| 2 | Uniform structured mesh, AMR hierarchy, halo fill, task-graph driver. |
| 3 | Platform services: manifest infrastructure, comparison-result schema, simulation spec format. |
| 4 | Visualization: unit-aware plotting, in-engine renderers, public gallery. |
| 5 | Newtonian hydrodynamics: finite-volume Godunov, PPM reconstruction, HLLC/HLLE Riemann solvers. |
| 6 | Self-gravity and N-body: multigrid Poisson, particle infrastructure, Barnes–Hut tree. |
| 7 | Microphysics: EOS interface, reaction networks, cooling tables, opacities. |
| 8 | MHD: ideal and resistive, constrained transport, super-time-stepping. |
| 9 | Radiation transport: gray FLD, multigroup FLD, two-moment M1. |
| 10 | Special and general relativity: SR hydro, GR hydro/MHD on fixed spacetimes, dynamical spacetime via BSSN. |
| 11 | Particle cosmology: SPH, meshless methods, FRW integrator, halo finders. |
| 12 | Moving mesh: Arepo-class Voronoi. *(stretch)* |
| 13 | Stellar evolution: 1-D Lagrangian solver with nuclear burning and mixing. *(stretch)* |
| 14 | Subgrid physics and synthetic observables: plugin interface, in-situ rendering. *(stretch)* |

---

## Platform & infrastructure

### Milestones

One-time deliveries that establish the platform foundation — covering process
discipline, documentation architecture, verification infrastructure, and
reproducibility tooling:

| Milestone | Capability |
|-----------|------------|
| M0 | Process discipline: branch/PR/commit/attribution standards. ✓ |
| M1 | Verification infrastructure: Function:/Source:/Sink: block convention, formulas register, convergence testing helpers, externally-grounded test pattern demonstrated on Epoch 1 kernel. ✓ |
| M2 | Documentation architecture: all live architectural decisions as one-paragraph claims in a single file; docs/ consolidated to index + API reference. |
| M3 | Validation infrastructure: manifests, provenance sidecars, and comparison-result schema. Planned alongside simulation Epoch 3. |
| M4 | Reproducibility capsule tooling: self-executing builder from the architectural basis established in M2. |
| M5 | Application-repo capsule integration and multi-repository evidence regeneration. |

### Per-epoch verification standard

Every simulation epoch must satisfy this checklist before it is considered verified:

- Function:/Source:/Sink: block on every operator class introduced
- Entry in the formulas register (`replication/formulas.md`) for each physics formula
- At least one externally-grounded test (analytical solution or published benchmark — not an engine-generated golden file)
- At least one convergence test confirming the stated approximation order p
- Lane A/B/C classification stated; derivation document with SymPy checks for Lanes B and C

---

## Immediate next work

### Simulation

1. Add `FlatManifold(PseudoRiemannianManifold)` to `theory/`
2. Add `geometry/` with `EuclideanSpace` and `MinkowskiSpace`
3. Thread `ndim` from manifold through `computation/` via `LocatedDiscretization.manifold` and a `Domain` type
4. Add `∂M` to `theory/`
5. Add `BoundaryCondition(Function)` ABC

### Infrastructure

1. M2: complete (architectural decisions consolidated into `ARCHITECTURE.md`; `docs/` consolidated to index + API reference)
2. Apply per-epoch verification standard to simulation Epoch 1 (convergence test for the Laplacian stencil)
