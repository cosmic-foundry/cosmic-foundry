# Cosmic Foundry — Roadmap

For cross-cutting architectural decisions, see [`ARCHITECTURE.md`](ARCHITECTURE.md).
For near-term work and current status, see [`STATUS.md`](STATUS.md).

This file covers the long-horizon capability sequence: epochs, milestones, and the
verification standard. Items belong here when they are not yet specified well enough
to implement without further design discussion. Once an item is fully specified and
unblocked, it moves to `STATUS.md`.

---

## Open architectural questions

**How do concrete fields connect to the theory layer?**
`theory/` defines the abstract interface (`ScalarField`, `VectorField`, etc.).
The concrete implementation — a JAX array carrying values at grid points — must
subclass those ABCs. The design question is how the concrete class declares its
grid, how indexing works, and whether field arithmetic lives on the concrete class
or on a separate operator. This must be resolved before Epoch 2.

**How does a grid relate to `Region`?**
A Cartesian grid is a `Region` — a compact connected subset of ℝⁿ — but it also
carries discrete structure (cell count, spacing, coordinate arrays). Whether the
grid IS a `Region` subclass or OWNS a `Region` is an open question. The answer
determines how boundary conditions attach and how the manifold hierarchy connects
to numerical infrastructure.

**Provenance on engine-written files.**
Every HDF5 file written by the engine should embed the git commit hash and
dirty-tree flag. Planned alongside Epoch 6 (I/O).

---

## Planned theory additions

**`DynamicManifold(PseudoRiemannianManifold)`**
— A manifold whose metric tensor is a dynamical field in the simulation state
rather than a structural property. Required for full GR (3+1 ADM formalism):
spatial hypersurfaces Σ_t are 3-D Riemannian; the 3-metric γ_ij and extrinsic
curvature K_ij are evolved fields. Planned alongside Epoch 14.

**`Connection` / `AffineConnection`**
— Covariant derivative; not a tensor field (inhomogeneous transformation law).
Required for curvature computations and parallel transport on curved spacetimes.
Planned alongside Epoch 14.

---

## Simulation capabilities

The epoch sequence below is a best-guess ordering as of the current foundation.
It should be sharpened in the next design session — in particular, the boundary
between Epochs 2–4 (concrete implementation of the abstract layer) and the
subsequent physics epochs.

| Epoch | Capability |
|-------|------------|
| 0 | Project scaffolding: CI, pre-commit, documentation standards. ✓ |
| 1 | Stencil derivation pipeline: SymPy-based finite-difference coefficients, arbitrary order, convergence tests. ✓ |
| 2 | Concrete geometry: `CartesianGrid` as a concrete `Region`; coordinate arrays; cell volumes and face areas. |
| 3 | Concrete fields: JAX-backed `ScalarField` and `VectorField` on a `CartesianGrid`; field arithmetic and pointwise operations. |
| 4 | Numerical operators: concrete `DifferentialOperator` subclasses (gradient, divergence, Laplacian) using the Epoch 1 stencil coefficients; convergence tests against analytical solutions. |
| 5 | Mesh infrastructure: ghost cells and halo fill; physical boundary condition application; structured-mesh topology. |
| 6 | I/O and diagnostics: HDF5 field output with provenance sidecars; checkpointing; simple time-series diagnostics. |
| 7 | Scalar transport: linear advection and diffusion; method-of-lines time integration; analytical convergence tests. |
| 8 | Newtonian hydrodynamics: Euler equations, finite-volume Godunov, PPM reconstruction, HLLC/HLLE Riemann solvers. |
| 9 | Self-gravity: multigrid Poisson solver; N-body particle infrastructure; Barnes–Hut tree. |
| 10 | Microphysics: EOS interface, reaction networks, cooling tables, opacities. |
| 11 | MHD: ideal and resistive, constrained transport, super-time-stepping. |
| 12 | Radiation transport: gray FLD, multigroup FLD, two-moment M1. |
| 13 | AMR: adaptive mesh refinement hierarchy, coarse–fine interpolation, load balancing. |
| 14 | Special and general relativity: SR hydro, GR hydro/MHD on fixed spacetimes, dynamical spacetime via BSSN. |
| 15 | Particle cosmology: SPH, meshless methods, FRW integrator, halo finders. *(stretch)* |
| 16 | Moving mesh: Arepo-class Voronoi tessellation. *(stretch)* |
| 17 | Stellar evolution: 1-D Lagrangian solver with nuclear burning and mixing. *(stretch)* |
| 18 | Subgrid physics and synthetic observables: plugin interface, in-situ rendering. *(stretch)* |

---

## Platform & infrastructure

### Milestones

| Milestone | Capability |
|-----------|------------|
| M0 | Process discipline: branch/PR/commit/attribution standards. ✓ |
| M1 | Verification infrastructure: convergence testing helpers, formulas register, externally-grounded test pattern. ✓ |
| M2 | Documentation architecture: all live architectural decisions as one-paragraph claims in a single file; docs/ as a minimal index with API reference. ✓ |
| M3 | Validation infrastructure: manifests, provenance sidecars, comparison-result schema. Planned alongside Epoch 6. |
| M4 | Reproducibility capsule tooling: self-executing builder from the architectural basis. |
| M5 | Application-repo capsule integration and multi-repository evidence regeneration. |

### Per-epoch verification standard

Every simulation epoch must satisfy this checklist before it is considered verified:

- Derivation document with SymPy checks for any new numerical scheme (Lanes B and C)
- Entry in the formulas register (`replication/formulas.md`) for each physics formula
- At least one externally-grounded convergence test against an analytical solution
  or observational data (not an engine-generated golden file)
- Lane A/B/C classification stated in the PR description
