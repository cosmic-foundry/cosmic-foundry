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
| `cosmic_foundry/mesh/` | Spatial partitioning: uniform Cartesian patches, domain partition, halo fill. | `mesh/__init__.py` |
| `cosmic_foundry/io/` | Array I/O — write to HDF5, merge rank files. | `io/__init__.py` |
| `cosmic_foundry/observability/` | Structured logging. | `observability/__init__.py` |
| `cosmic_foundry/manifests/` | Manifest infrastructure — HTTP client, schema validation, provenance. | `manifests/__init__.py` |
| `cosmic_foundry/cli/` | CLI entry point (`cosmic-foundry`). | `cli/main.py` |
| `tests/` | Test suite. `tests/utils/` holds shared stencil and convergence helpers. | — |
| `benchmarks/` | Performance benchmarks (roofline, throughput). | — |
| `replication/` | Formula register and replication targets. | `replication/formulas.md` |
| `derivations/` | SymPy derivation documents for physics capabilities (Lane B/C). | — |
| `research/` | Research notes and reference surveys. Planned to migrate into `docs/` as rendered theory content once the Sphinx pipeline is mature. | — |
| `pr-review/` | Adversarial PR review checklist and architecture stress-review checklist. | `pr-review/README.md` |
| `scripts/` | Agent health check, PR review wrappers, session startup. | `scripts/agent_health_check.sh` |
| `environment/` | Conda environment setup and activation. | `environment/setup_environment.sh` |

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

### `cosmic_foundry/geometry/`

Concrete simulation geometry classes — the first real objects
implementing the `theory/` manifold ABC chain. Planned contents:

**`FlatManifold(PseudoRiemannianManifold)`** *(first goes in `theory/`)*
— A pseudo-Riemannian manifold with zero Riemann curvature tensor.
Branches from `PseudoRiemannianManifold` (not `RiemannianManifold`) so
that both Euclidean and Minkowski spaces can inherit from it.

**`EuclideanSpace(RiemannianManifold, FlatManifold)`**
— ℝⁿ with the standard flat positive-definite metric. Only free
parameter: `n: int`. `signature = (n, 0)` and `ndim = n` are both
derived.

**`MinkowskiSpace(FlatManifold)`**
— ℝ⁴ with Lorentzian signature `(1, 3)`. Flat pseudo-Riemannian
background for special-relativistic simulations. No free parameters.

**`Domain`**
— A manifold equipped with physical bounds and topology:
`manifold: SmoothManifold`, `origin: tuple[float, ...]`,
`size: tuple[float, ...]`. Replaces the raw keyword arguments currently
passed to `PartitionDomain.execute`. `Domain.manifold.ndim` is the
source of truth for dimensionality throughout the computation stack.

### Planned `theory/` additions

**`FlatManifold(PseudoRiemannianManifold)`** — see above; goes in
`theory/` before `geometry/` is created.

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

### Threading `ndim` through `computation/`

`SmoothManifold.ndim` exists. It is not yet threaded to `Patch`,
`Stencil`, or `PartitionDomain`. Planned: `LocatedDiscretization`
declares an abstract `manifold` property; `Patch` stores the manifold
and derives `ndim` from `manifold.ndim`; `PartitionDomain.execute`
takes a `Domain` (above) as input. `Stencil` validates that
`len(radii) == manifold.ndim` at construction.

---

## Current work

Immediate code work (in dependency order):

1. Add `FlatManifold` to `theory/`
2. Create `geometry/` with `EuclideanSpace` and `MinkowskiSpace`
3. Thread `ndim` from manifold through `computation/`
4. Add `∂M` to `theory/`
5. Add `BoundaryCondition` ABC
