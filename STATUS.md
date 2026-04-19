# Cosmic Foundry — Status

This file is the navigation anchor for the repository. It tells you
where to find the authoritative description of each part of the codebase
and describes planned modules that do not yet have code.

For cross-cutting architectural decisions and open design questions, see
[`ARCHITECTURE.md`](ARCHITECTURE.md).
For development workflow and contribution process, see
[`DEVELOPMENT.md`](DEVELOPMENT.md).
For the high-level capability roadmap, see [`ROADMAP.md`](ROADMAP.md).

---

## How to read this repo

The code is the authoritative architecture description. Start here:

| Directory | What it is | Read first |
|---|---|---|
| `cosmic_foundry/theory/` | Pure mathematical ABCs — sets, manifolds, discretizations, functions, fields. No JAX dependency. | `theory/__init__.py` |
| `cosmic_foundry/computation/` | Distance-1 concrete implementations of theory ABCs. JAX-backed. | `computation/__init__.py` |
| `cosmic_foundry/geometry/` | Concrete manifolds and simulation domains: `EuclideanSpace`, `MinkowskiSpace`, `Domain`. | `geometry/__init__.py` |
| `cosmic_foundry/mesh/` | Spatial partitioning: uniform Cartesian patches, domain partition, halo fill. | `mesh/__init__.py` |
| `cosmic_foundry/io/` | Array I/O — write to HDF5, merge rank files. | `io/__init__.py` |
| `cosmic_foundry/observability/` | Structured logging. | `observability/__init__.py` |
| `cosmic_foundry/manifests/` | Manifest infrastructure — HTTP client, schema validation, provenance. | `manifests/__init__.py` |
| `cosmic_foundry/cli/` | CLI entry point (`cosmic-foundry`). | `cli/main.py` |
| `tests/` | Test suite. `tests/utils/` holds shared stencil and convergence helpers. | — |
| `benchmarks/` | Performance benchmarks (roofline, throughput). | — |
| `replication/` | Formula register and replication targets. | `replication/formulas.md` |
| `derivations/` | SymPy derivation documents for physics capabilities (Lane B/C). | — |
| `docs/research/` | Research survey — code landscape, capabilities, licensing, V&V methodology. Lives in `docs/` but not yet woven into the Sphinx site; the goal is conceptual integration (cross-links, rendered pages) not just co-location. | `docs/research/index.md` |
| `pr-review/` | Adversarial PR review checklist and architecture stress-review checklist. | `pr-review/README.md` |
| `scripts/` | Agent health check, PR review wrappers, session startup, environment setup and activation. | `scripts/agent_health_check.sh` |
| `environment/` | Conda environment spec files and miniforge install target. | `environment/cosmic_foundry.yml` |

### The mathematical hierarchy at a glance

`theory/` defines an ABC tree; `computation/` and `mesh/` implement it
at distance 1. The full hierarchy is in `ARCHITECTURE.md §Mathematical
hierarchy`. The three root ABCs and their concrete children:

- **`IndexedFamily`** → `Array[T]` *(computation/)*
- **`IndexedSet`** → `Extent` *(computation/)* → `Patch` *(mesh/)*
- **`Function`** → `Stencil`, `Reduction` *(computation/)*, `PartitionDomain` *(mesh/)*

---

## Planned modules

These modules are designed but do not yet have code. The descriptions
here are the authoritative record until code exists.

### Planned `theory/` additions

**`∂M` (manifold boundary)**
— An operation or property on `SmoothManifold` returning the boundary
manifold `∂M`, which has dimension `ndim - 1`. Needed to formally type
`BoundaryCondition` (below). Concrete discrete form: the set of index
faces at the boundary of an `Extent`.

**`DynamicManifold(PseudoRiemannianManifold)`**
— A manifold whose signature is fixed but whose metric tensor is a
dynamical field in the simulation state rather than a structural
property. Required for full GR simulations. In the 3+1 (ADM) formalism:
spatial hypersurfaces Σ_t are 3-D Riemannian; the 3-metric `γ_ij` and
extrinsic curvature `K_ij` are evolved fields.

### Planned `computation/` and `theory/` additions

**`BoundaryCondition(Function)`** *(in `theory/`)*
— A function that operates on `∂M`-indexed data and enforces a
condition on field values at the boundary. Blocked on `∂M` existing so
the codimension-1 invariant is enforced at the ABC level. Concrete
subclasses (`DirichletBC`, `NeumannBC`, `PeriodicBC`) live in
`computation/`.

---

## Current work

Immediate code work (in dependency order):

1. Move `EuclideanSpace` and `MinkowskiSpace` to `theory/`; make `Domain` a `ManifoldWithBoundary`
2. Add `BoundaryCondition` ABC
