# Cosmic Foundry — Roadmap

For cross-cutting architectural decisions, see [`ARCHITECTURE.md`](ARCHITECTURE.md).
For the current codebase state and planned modules, see [`STATUS.md`](STATUS.md).

---

## Simulation capabilities

| Epoch | Capability |
|-------|------------|
| 0 | Project scaffolding: CI, pre-commit, ADR process, entry point. ✓ |
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

## V&V capabilities

| Epoch | Capability |
|-------|------------|
| M0 | Branch, PR, commit-size, history, and attribution discipline. ✓ |
| M1 | Capability specs, formulas register, and externally-grounded tests for each implemented map. ✓ |
| M2 | Minimal architectural basis: all live architectural decisions expressed as a complete, orthogonal set of one-paragraph claims in a single file. |
| M3 | Convergence coverage: each implemented physics map has at least one MMS or analytical convergence test confirming its stated approximation order. |
| M4 | Lane A/B/C derivation documents with SymPy checks, accompanying each physics capability as it is implemented. |
| M5 | Validation manifests, provenance sidecars, and comparison-result schema. Planned alongside simulation Epoch 3. |
| M6 | Reproducibility capsule tooling: self-executing builder from the architectural basis established in M2. |
| M7 | Application-repo capsule integration and multi-repository evidence regeneration. |

---

## Immediate next work

### Simulation

1. Add `FlatManifold(PseudoRiemannianManifold)` to `theory/`
2. Add `geometry/` with `EuclideanSpace` and `MinkowskiSpace`
3. Thread `ndim` from manifold through `computation/` via `LocatedDiscretization.manifold` and a `Domain` type
4. Add `∂M` to `theory/`
5. Add `BoundaryCondition(Function)` ABC

### V&V

1. M3: convergence coverage for each currently implemented physics map
