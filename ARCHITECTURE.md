# Cosmic Foundry — Architecture

## Commitments

These are the foundational claims about this repository. Each is a
commitment: the code must satisfy it, tests enforce it where possible,
and any PR that violates a claim must explicitly revise it here rather
than quietly breaking it.

**Cosmic Foundry is a general-purpose PDE simulation engine, optimized
for astrophysical use cases.**

**The mathematical language of the architecture is differential geometry
on spatio-temporal manifolds, with PDE theory as the application layer.**

**Physical quantities are represented as instances of formal mathematical
abstractions.**

**Every numerical method is formally derived from its continuous
mathematical counterpart, and machine-checkable, where possible.**

**Every object that claims to represent a specific physical scenario is paired
with a testable claim about that scenario — symbolic or numerical — that CI can
check. For numerical methods this claim is verification against an analytical
solution, with the test living in this repository.**

**Where external data sources are ingested the uncertainty in that
data is explicitly quantified and propagated.**

**The engine is dimensionless internally.**

---

## Layer architecture

The codebase is organized into four packages with a strict dependency order:

```
foundation/   ←  continuous/
     ↑                ↑
     └──────── discrete/ (Epoch 2, not yet implemented)
                        ↑
                  computation/
```

**`foundation/` and `continuous/` are symbolic-reasoning layers.**
Their shared identity: they describe mathematical structure symbolically, without
numerical evaluation. Their import boundary reflects that identity — they may
only import from the Python standard library, `cosmic_foundry`, or packages on
the approved symbolic-reasoning list. The approved list is `{sympy}`. Additions
require justification against the symbolic-reasoning identity; numerical
computation packages (JAX, NumPy, SciPy) are excluded by definition. Enforced
by `tests/test_theory_no_third_party_imports.py`.

### foundation/  · Epoch 1 ✓

```
Set
├── TopologicalSpace     — Set equipped with a topology (marker; no additional interface)
├── IndexedFamily        — finite collection indexed by {0,…,n-1}; interface: __getitem__, __len__
└── IndexedSet           — finite rectangular subset of ℤⁿ; interface: shape, intersect
                           derived: ndim = len(shape)

Function[D, C]           — callable mapping domain D → codomain C; interface: __call__
├── SymbolicFunction     — Function defined by a SymPy expression; free: expr, symbols
│                          derived: __call__ = expr.subs(zip(symbols, args))
├── NumericFunction      — Function implemented procedurally; interface: __call__
│                          optional: symbolic → SymbolicFunction (refinement declaration)
└── InvertibleFunction   — bijection with two-sided inverse; interface: domain, codomain, inverse
    └── Homeomorphism    — bicontinuous bijection; narrows domain/codomain to TopologicalSpace
```

### continuous/  · Epoch 1 ✓

```
TopologicalManifold(TopologicalSpace) — locally Euclidean topological space; interface: ndim
└── Manifold                          — TopologicalManifold + smooth atlas; interface: atlas → Atlas
    └── PseudoRiemannianManifold      — Manifold + metric; free: signature, metric
                                        derived: ndim = sum(signature)
        └── RiemannianManifold        — positive-definite metric; free: ndim, metric
                                        derived: signature = (ndim, 0)

Diffeomorphism(Homeomorphism)         — smooth bijection; narrows domain/codomain to Manifold
└── Chart                             — local coordinate system φ: U → V; co-located in manifold.py

Atlas(IndexedFamily)                  — collection of Charts covering M; co-located in manifold.py
                                        interface: __getitem__ → Chart, __len__

MetricTensor(SymmetricTensorField)    — metric g; co-located in pseudo_riemannian_manifold.py

Field(SymbolicFunction)               — f: M → V; interface: manifold → Manifold, expr, symbols
└── TensorField                       — interface: tensor_type → (p, q)
    ├── SymmetricTensorField          — derived: tensor_type = (0, 2); interface: component(i,j) → Field
    │   └── MetricTensor             — see above
    └── DifferentialForm             — free: degree; derived: tensor_type = (0, degree)

DifferentialOperator(Function[Field, Field]) — L: Field → Field; interface: manifold, order

Constraint(ABC)                       — interface: support → Manifold
└── BoundaryCondition                 — support is ∂M
    ├── LocalBoundaryCondition        — α·f + β·∂f/∂n = g; free: alpha, beta, constraint
                                        derived: support = constraint.manifold
    └── NonLocalBoundaryCondition     — constraint depends on values outside the immediate neighborhood
```

**`Constraint` / `BoundaryCondition` hierarchy.** `LocalBoundaryCondition`
covers Dirichlet (`α=1, β=0`), Neumann (`α=0, β=1`), and Robin via the
unified `α·f + β·∂f/∂n = g` form. `NonLocalBoundaryCondition` makes no
claim about the form of the non-locality; concrete subclasses declare
whatever geometric references they need.

**Class existence is justified by a falsifiable constraint, not anticipation.**
Every ABC in `continuous/` and `foundation/` must earn its place. A new class
is warranted only when it removes a degree of freedom from its parent: either
a derived property (a non-abstract property fully determined by abstract ones)
or a type narrowing that mypy can check. A property describing *regularity* of
`__call__` (continuous, smooth) is not falsifiable in Python's type system and
does not justify a new class.

Concretely: `PseudoRiemannianManifold` earns its place via `ndim =
sum(signature)`; `RiemannianManifold` via `signature = (ndim, 0)`;
`Homeomorphism` by narrowing `domain`/`codomain` to `TopologicalSpace`. A
hypothetical `SmoothMap` would not qualify.

Non-independent objects are co-located in the same file as the object they
belong to. Example: `Chart`, `Atlas`, `Diffeomorphism` in `manifold.py`;
`MetricTensor`, `RiemannianManifold` in `pseudo_riemannian_manifold.py`.

**Planned additions** (Epoch 12)

**`DynamicManifold(PseudoRiemannianManifold)`** — A manifold whose metric
tensor is a dynamical field in the simulation state. Required for full GR
(3+1 ADM formalism): signature is fixed (Lorentzian), but the metric is
evolved by the Einstein equations. In the 3+1 decomposition the
computational domain is a 3-D Riemannian spatial hypersurface; the
3-metric `γ_ij` and extrinsic curvature `K_ij` are evolved fields. The
concrete entry would be `Spacetime3Plus1(DynamicManifold)`. Interface not
yet designed.

**`Connection` / `AffineConnection`** — Covariant derivative; not a tensor
field (inhomogeneous transformation law). Required for curvature
computations and parallel transport.

**Open questions**

**What is the formal PDE object in the continuous layer?**
The primary PDE form targeted by the discrete layer is the conservation law in
divergence form: ∂ₜU + ∇·F(U) = S. The integral form —
∫_Ωᵢ ∂ₜU dV + ∮_∂Ωᵢ F·n dA = ∫_Ωᵢ S dV — is what `discrete/` approximates.
Whether this warrants a formal `ConservationLaw` class in `continuous/` — earning
its place if it can derive the weak form (for FEM) and the flux-divergence form
(for FVM) as derived properties — is unsettled.

**What do SymPy-backed continuous objects look like?**
The open case is coordinate-dependent fields: a concrete `ScalarField` backed
by a SymPy expression `f(x, y) = sin(πx)sin(πy)` where the coordinate symbols
`x, y` are tied to a specific chart. The interface for coordinate-dependent
SymPy-backed fields (evaluatable analytical forms, coordinate-to-chart binding)
is not yet designed. Concrete field implementations live outside `continuous/`
— either in test fixtures or in `computation/` once the numerical layer lands.

### discrete/  · Epochs 2–3

The discrete layer approximates the **integral form** of conservation laws, not
the differential form. The derivation chain grounding every object in this layer:

1. A conservation law in divergence form on a domain Ω ⊂ M: ∂ₜU + ∇·F(U) = S
2. Integrate over each control volume Ωᵢ and apply the divergence theorem:
   ∂ₜ∫_Ωᵢ U dV + ∮_∂Ωᵢ F·n dA = ∫_Ωᵢ S dV
3. Approximate cell averages Ūᵢ ≈ |Ωᵢ|⁻¹ ∫_Ωᵢ U dV and face fluxes at each
   shared interface; this yields the discrete scheme

Finite volume (FVM) is the primary method — every term has a geometric
interpretation (cell volume, face area, face normal) derived from the chart and
the cell decomposition. FDM and FEM are also derivable from this foundation:

- **FDM**: On a Cartesian mesh with midpoint quadrature and piecewise-constant
  reconstruction, FVM reduces to FDM. Finite difference is a special case of
  FVM on regular meshes, not a separate derivation.
- **FEM**: Multiplying by a test function and integrating by parts yields the
  weak formulation; choosing a finite-dimensional function space Vₕ yields FEM.
  Additional machinery (basis functions, bilinear forms, function spaces) extends
  the current foundation; deferred.

The earlier `DiscreteField` ABC was removed: it predated the `Chart`/`Atlas`
machinery and baked the approximation relationship (which involves three
parties — continuous field, mesh, discretization scheme) into the discrete
object alone.

**Planned** (Epoch 2):

```
Mesh                        — cell decomposition of a domain Ω ⊂ M; primary abstract object
  cells                     — n-dimensional control volumes
                              interface: volume, center
  faces                     — (n-1)-dimensional interfaces between adjacent cells
                              interface: area, normal, neighboring cells (or boundary flag)
  connectivity              — which cells share which faces

[name TBD](Mesh)            — a Mesh whose cells are regular and axis-aligned; carries
                              chart: Chart grounding coordinate symbols symbolically.
                              abstract: coordinate(idx) → ℝⁿ (values in chart's codomain)
                              evaluation bridge:
                                field.expr.subs(zip(chart.symbols, coordinate(idx)))
                              The integer index structure (shape, ndim, intersect) is a
                              derived consequence of cell regularity — this class satisfies
                              the IndexedSet interface without descending from it.

CartesianMesh([name TBD])   — concrete; free: origin, spacing, shape
                              derives: coordinate, cell volumes, face areas, face normals

MeshFunction(NumericFunction[Mesh, V])
                            — value assignment to mesh elements (cells, faces, or vertices)
                              earns its class via .mesh: Mesh typed accessor,
                              by analogy with Field.manifold
```

**Planned** (Epoch 3):

```
DiscreteOperator(NumericFunction[MeshFunction, MeshFunction])
                            — derived from a continuous DifferentialOperator via the
                              divergence theorem applied per cell; not independently
                              constructed from stencil coefficients.
                              Truncation error verified algebraically via SymPy.
```

**Open questions**

**What is the name for the regular-mesh ABC?**
The structured-mesh ABC sitting between `Mesh` and `CartesianMesh` needs a
name. Candidates: `StructuredMesh` (standard CFD terminology), `Grid`
(conventional but implies index-first derivation rather than control-volume-first),
`RegularMesh` (accurate but generic). Name is TBD; the design is settled.

**What is the `Mesh` ABC?**
Whether `Mesh` is a formal ABC under the falsifiable-constraint rule — or whether
cell/face structure is always concretely realized — is unsettled. `Mesh` earns
its place if unstructured meshes (Epoch 14, moving mesh) are in scope; for
Epochs 2–11 the structured concrete classes suffice.

**Is scheme choice a first-class concept?**
A discretization is a precise mathematical act: choose a mesh, choose an
approximation order, derive face fluxes from the divergence theorem. Whether a
formal `Discretization` — a callable mapping a conservation law + mesh +
approximation order to a discrete scheme — belongs in `discrete/`, or whether
scheme choice remains implicit in how `DiscreteOperator` is constructed,
is unsettled.

### computation/  · Epoch 4

JAX evaluation. The only layer that touches floats. Planned: concrete field
storage as `jax.Array`; JIT-compiled stencil application; explicit time
integration; HDF5 I/O with provenance.

**Open question**

**Kernel composition model.**
A backend-agnostic interface separating kernel computation (Op) from
spatial domain and execution policy (Policy) is a design goal. An
earlier Op/Policy/Dispatch framing was dropped before it was realized.
The formal model governing composition, backend substitutability, and
dispatch is unsettled.

### Cross-cutting

**Numerical transcription discipline.**
Physics capabilities sourced from reference tables (EOS polynomial fits,
reaction networks, opacity tables) need a discipline governing how
numeric tables are transcribed, verified, and updated independently of
the derivation-first lane policy. This decision is deferred to Epoch 7
(microphysics), when the first such capability lands.

**Physical constants ingestion (CODATA).**
The engine will need physical constants (G, c, ħ, k_B, …) throughout the
physics epochs. The authoritative machine-readable source is NIST CODATA
(public domain), available at `https://physics.nist.gov/cuu/Constants/Table/allascii.txt`.
Open questions: where the constants module lives (`foundation/`? `computation/`?)
and whether it must respect the symbolic-reasoning import boundary; how constants
are exposed (SymPy symbols with known numerical values, plain floats, or both);
how the CODATA revision is pinned and updated. WGS 84 / GPS-specific defined
constants (μ, Ω_E, GPS semi-major axis) have no machine-readable API; the
ingestion discipline for PDF-sourced defined constants is a separate decision.

---

## Current work

**M3: Executable mathematical narrative.**
The first `validation/` implementations are GR spacetimes: `SchwarzschildSpacetime`
with a SymPy-backed metric, GPS time dilation derivation, and Schwarzschild
embedding diagram. Each is a concrete `PseudoRiemannianManifold` alongside
SymPy assertions that CI executes. Notebooks in `docs/` import directly from
`validation/` — no class definitions in notebooks. Open questions to settle
during implementation: coordinate-to-chart binding (which SymPy symbols belong
to which `Chart`), and how `symbols` is declared on concrete `Field` subclasses.

**Epoch 2 design decision: the regular-mesh ABC carries its Chart.**
The structured-mesh ABC ([name TBD]) carries an abstract `chart: Chart`
property. `coordinate(idx)` returns values in the codomain of that chart.
The chart's symbols are the symbolic names for the coordinates; the evaluation
bridge `field.expr.subs(zip(mesh.chart.symbols, mesh.coordinate(idx)))` connects
continuous field expressions to discrete mesh values. For `CartesianMesh`,
`coordinate` is derived from `origin`, `spacing`, and the multi-index.
The integer index structure is a derived consequence of cell regularity, not
an inherited ancestor.

---

## Physics roadmap

Each physics epoch adds new fields and equations to the continuous layer and
extends the discrete and numerical layers minimally to evaluate them.

### Foundation epochs

| Epoch | Layer | Capability |
|-------|-------|------------|
| 0 | — | Project scaffolding: CI, pre-commit, documentation standards. ✓ |
| 1 | Continuous | `continuous/` ABCs: full manifold and field hierarchy, operators, boundary conditions, metric; coordinate structure (`Chart`, `Atlas`). `foundation/` ABCs: `Set`, `Function`, `IndexedSet`, `IndexedFamily`. ✓ |
| 2 | Discrete | `Mesh` ABC (cell decomposition: cells, faces, connectivity); regular-mesh ABC [name TBD] with `chart: Chart` and abstract `coordinate`; `CartesianMesh` concrete (derives cell volumes, face areas, face normals); `MeshFunction` with `.mesh` accessor. Integer index structure derived from cell regularity, not inherited. FVM as primary discretization method. |
| 3 | Discrete | `DiscreteOperator` derived from continuous `DifferentialOperator` via the divergence theorem applied per cell; truncation error verified algebraically via SymPy. First working Poisson solver on `CartesianMesh`. |
| 4 | Numerical | JAX evaluation layer: concrete field storage as `jax.Array`; JIT-compiled stencil application; explicit time integration; HDF5 I/O with provenance. |

### Physics epochs

| Epoch | Capability |
|-------|------------|
| 5 | Scalar transport: linear advection and diffusion on a `CartesianMesh` via FVM. First end-to-end simulation; validates the full pipeline. |
| 6 | Newtonian hydrodynamics: Euler equations, FVM Godunov, PPM reconstruction, HLLC/HLLE Riemann solvers. |
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

## Platform milestones

| Milestone | Capability |
|-----------|------------|
| M0 | Process discipline: branch/PR/commit/attribution standards. ✓ |
| M1 | Verification infrastructure: convergence testing helpers, externally-grounded test pattern. ✓ |
| M2 | Documentation architecture: all live architectural decisions in `ARCHITECTURE.md`; `docs/` as API reference index. ✓ |
| M3 | Executable mathematical narrative: first `validation/` implementations (Schwarzschild spacetime, GPS time dilation); notebooks in `docs/` that import from `validation/` and run in CI. Settles coordinate-to-chart binding and the `SymbolicFunction` interface on concrete fields. ✓ |
| M4 | Validation infrastructure: manifests, provenance sidecars, comparison-result schema. Planned alongside Epoch 4. |
| M5 | Reproducibility capsule tooling: self-executing builder. |
| M6 | Application-repo capsule integration and multi-repository evidence regeneration. |

### Per-epoch verification standard

Every physics epoch must satisfy this checklist before it is considered verified:

- Derivation document with SymPy checks for any new numerical scheme (Lanes B and C)
- At least one externally-grounded convergence test against an analytical solution
  or observational data (not an engine-generated golden file); where an analytical
  solution exists, the relevant `NumericFunction.symbolic` is declared so the
  check runs automatically
- Lane A/B/C classification stated in the PR description
