# Cosmic Foundry — Roadmap

For cross-cutting architectural decisions, see [`ARCHITECTURE.md`](ARCHITECTURE.md).
For near-term work — planned modules and the immediate implementation queue — see [`STATUS.md`](STATUS.md).

This file covers the long-horizon capability sequence: epochs, milestones, and the
verification standard. Items belong here when they are not yet specified well enough
to implement without further design discussion. Once an item is fully specified and
unblocked, it moves to `STATUS.md`.

---

## Architectural basis gap closure

Items required to bring the codebase into consistency with the
foundational claims in `ARCHITECTURE.md §Architectural basis`.

**Wire `Field` instances to the computation layer** *(claim 4).*
Kernel inputs and outputs are currently raw JAX arrays wrapped in
`Array[T]`. The design intention is that physical quantities are
`Field` instances. Requires a design decision on how `Field` subclasses
connect to `Array[T]` and how kernels declare their field types.

**Create `derivations/` and add a Laplacian stencil derivation**
*(claim 5).* The `derivations/` directory does not exist. The
second-order 7-point Laplacian stencil needs a SymPy derivation
document demonstrating that the finite-difference weights are the
correct Taylor-series approximation of ∂²/∂x².

**Apply convergence test to the Laplacian stencil** *(claim 6).*
The convergence infrastructure in `tests/utils/convergence.py` exists
but is not applied to any production operator. The Laplacian must be
shown to converge at second order under grid refinement against an
analytical solution (e.g. f(x) = sin(2πx), ∇²f = -(2π)²sin(2πx)).

**Attach provenance metadata to all engine-written files** *(claim 9).*
`io/` writes HDF5 files with no git commit hash. Every call to
`WriteArray.execute()` should embed the current repository state
(git commit hash, dirty-tree flag) as HDF5 attributes. Planned
alongside M3 (validation infrastructure).

---

## Planned visualization stack

Field data is written in HDF5 (current `io/`) and Zarr v3 (planned).
Browser rendering uses WebGPU primary with a WebGL2 fallback; desktop
rendering uses pyvista/vispy for local inspection. All colormaps are
perceptual (cmasher, cmocean); rainbow/jet are prohibited. Visual
regression tests use pytest-mpl with SSIM comparison.

---

## Planned theory additions

**`DynamicManifold(PseudoRiemannianManifold)`**
— A manifold whose signature is fixed but whose metric tensor is a
dynamical field in the simulation state rather than a structural
property. Required for full GR simulations. In the 3+1 (ADM) formalism:
spatial hypersurfaces Σ_t are 3-D Riemannian; the 3-metric `γ_ij` and
extrinsic curvature `K_ij` are evolved fields.

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
| M1 | Verification infrastructure: Function:/Source:/Sink: block convention, formulas register, convergence testing helpers, externally-grounded test pattern. ✓ |
| M2 | Documentation architecture: all live architectural decisions as one-paragraph claims in a single file; docs/ as a minimal index with API reference. ✓ |
| M3 | Validation infrastructure: manifests, provenance sidecars (git commit hash on every engine-written file), and comparison-result schema. Planned alongside simulation Epoch 3. |
| M4 | Reproducibility capsule tooling: self-executing builder from the architectural basis established in M2. |
| M5 | Application-repo capsule integration and multi-repository evidence regeneration. |

### Per-epoch verification standard

Every simulation epoch must satisfy this checklist before it is considered verified:

- Function:/Source:/Sink: block on every operator class introduced
- Entry in the formulas register (`replication/formulas.md`) for each physics formula
- At least one externally-grounded test (analytical solution or published benchmark — not an engine-generated golden file)
- At least one convergence test confirming the stated approximation order p
- Lane A/B/C classification stated; derivation document with SymPy checks for Lanes B and C
