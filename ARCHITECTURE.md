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
theory/
  foundation/   ←  continuous/
       ↑                ↑
       └──────── discrete/
                     ↑
geometry/   ← concrete instantiable objects (meshes, spacetimes)
    ↑
computation/
```

`foundation/`, `continuous/`, and `discrete/` are nested under `theory/`,
making the symbolic-reasoning boundary a directory boundary. Everything
outside `theory/` (`geometry/`, `computation/`, `validation/`) is the
application/concreteness layer.

**`theory/` and `geometry/` are the symbolic-reasoning layer.**
`foundation/`, `continuous/`, `discrete/`, and `geometry/` all share the same
identity: they describe mathematical structure symbolically, without numerical
evaluation. `geometry/` is coordinate geometry infrastructure — manifolds,
charts, and meshes defined by SymPy expressions; numerical array allocation
belongs in `computation/`. Their import boundary reflects that shared identity —
they may only import from the Python standard library, `cosmic_foundry`, or
packages on the approved symbolic-reasoning list. The approved list is
`{sympy}`. Additions require justification against the symbolic-reasoning
identity; numerical computation packages (JAX, NumPy, SciPy) are excluded by
definition. Enforced by `tests/test_theory_no_third_party_imports.py`.

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

**Planned additions** (Epoch 2)

**`ConservationLaw(DifferentialOperator)`** — A differential operator in
divergence form: the spatial operator `∇·F(U)` in `∂ₜU + ∇·F(U) = S`.
Free: `flux: Function[Field, TensorField]` (the F in ∇·F(U)) and
`source: Field` (the S). Earned by the derived integral form — per cell,
the divergence theorem gives `∮_∂Ωᵢ F(U)·n dA = ∫_Ωᵢ S dV` — which is
fully determined by `flux` and the divergence theorem and cannot be derived
from a bare `DifferentialOperator`.
`ConservationLaw` is spatial only: `∂ₜ` is handled by the time integrator
(Epoch 4), not this object. This separation is preserved under the 3+1 ADM
decomposition: in GR, covariant equations `∇_μ F^μ = S` decompose to
`∂ₜ(√γ U) + ∂ᵢ(√γ Fⁱ) = √γ S(α, β, γᵢⱼ, Kᵢⱼ)` — still a spatial
divergence operator with metric factors entering through the `Chart` and
curvature terms in `source`. `ConservationLaw` is stable through Epoch 12.

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
CellComplex(IndexedFamily)  — chain (C_*, ∂): complex[k] returns the Set of k-cells.
                              Adds boundary operators ∂_k: C_k → C_{k-1}.
                              Earned by ∂² = 0 (∂_{k-1} ∘ ∂_k = 0), the algebraic
                              identity underlying the divergence theorem.
                              Example — 2D Cartesian N×M grid:
                                C_0: (N+1)(M+1) vertices
                                C_1: N(M+1) horizontal + (N+1)M vertical edges
                                C_2: N×M cells
                                ∂₁: signed vertex-incidence; ∂₂: signed edge-incidence

Mesh(CellComplex)           — CellComplex carrying a Chart from continuous/.
                              The chart's metric makes the complex geometric.
                              Faces are the geometric primitives: each face is a region
                              in the chart's parameter space; cell volumes are derived
                              from face geometry via the divergence theorem:
                                |Ωᵢ| = (1/n) ∑_{f ∈ ∂Ωᵢ} xf · nf Af
                              General volumes and areas computed as ∫ √|g| dV and
                              ∫ √|g_σ| dA using the chart's metric g.
                              Earned by: volume, area, normal are derived properties
                              fully determined by the CellComplex and the Chart.
                              Covers: Cartesian (g = I), cylindrical (√|g| = r),
                              GR spacetimes (curved g), moving mesh (time-varying Chart).

Rₕ(NumericFunction[Function[M,V], MeshFunction])
                            — free: mesh: Mesh
                              restriction operator: (Rₕ f)ᵢ = |Ωᵢ|⁻¹ ∫_Ωᵢ f dV
                              The output MeshFunction has .mesh == Rₕ.mesh by
                              construction — the cell averages are indexed by the
                              cells of Rₕ.mesh and can live on no other mesh.
                              This is the formal bridge from continuous/ to discrete/:
                              a continuous Function plus a Mesh yields a MeshFunction.
                              When f is a Field (SymbolicFunction), the integral is
                              computed analytically via SymPy. Rₕ is what defines the
                              relationship between Field and MeshFunction; this is why
                              the earlier DiscreteField ABC was wrong — the restriction
                              depends on both the field and the mesh, not either alone.

StructuredMesh(Mesh)            — a Mesh whose cells are regular and axis-aligned; carries
                              chart: Chart grounding coordinate symbols symbolically.
                              abstract: coordinate(idx) → ℝⁿ (values in chart's codomain)
                              evaluation bridge:
                                field.expr.subs(zip(chart.symbols, coordinate(idx)))
                              Narrows complex[n] from Set to IndexedSet: the regularity
                              constraint implies the top-dimensional cells biject with a
                              rectangular region of ℤⁿ, earning shape, ndim, and intersect
                              as derived properties of cell regularity.

CartesianMesh(StructuredMesh)   — free: origin, spacing, shape; flat metric (g = I)
                              chart is derived internally: a Cartesian mesh has a
                              Cartesian chart by definition (EuclideanManifold(ndim).atlas[0])
                              derives: coordinate = origin + (idx + ½)·spacing
                                       cell volume = ∏ Δxₖ
                                       face area = ∏_{k≠j} Δxₖ  (face ⊥ to axis j)
                                       face normal = ê_j
                              Lives in geometry/, not discrete/.

MeshFunction(NumericFunction[Mesh, V])
                            — value assignment to mesh elements (cells, faces, or vertices);
                              earns its class via .mesh: Mesh typed accessor,
                              by analogy with Field.manifold
```

**Planned** (Epoch 3):

```
Discretization(NumericFunction[ConservationLaw, DiscreteOperator])
                            — free: law: ConservationLaw, mesh: Mesh, order: int
                              maps a ConservationLaw + Mesh + approximation order
                              to a DiscreteOperator; encapsulates the scheme choice.
                              Defined by the commutation diagram:
                                Lₕ ∘ Rₕ ≈ Rₕ ∘ L   (up to O(hᵖ))
                              where p is the approximation order. Different scheme
                              choices (reconstruction, Riemann solver) are different
                              ways of constructing Lₕ to make the diagram commute at
                              order p. The _derive() function required by Lanes B and C
                              IS this commutation check, verified algebraically via SymPy.
                              Formally separate from Rₕ: Rₕ projects field values
                              (Function → MeshFunction); Discretization projects
                              operators (ConservationLaw → DiscreteOperator).
                              They share a Mesh parameter and are related by the
                              commutation diagram, but operate on different objects.

DiscreteOperator(NumericFunction[MeshFunction, MeshFunction])
                            — the output of Discretization; the Lₕ that makes
                              Lₕ ∘ Rₕ ≈ Rₕ ∘ L hold to the chosen order.
                              Earns its class via .mesh: Mesh — constrains input and
                              output to the same mesh (operator.mesh == input.mesh ==
                              output.mesh), by analogy with DifferentialOperator.manifold.
                              Not independently constructed from stencil coefficients.
```

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
the derivation-first lane policy. This decision is deferred to Epoch 9
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

**Epoch 2 design decisions: the discrete layer hierarchy.**
`CellComplex(IndexedFamily)` is the topological skeleton: `complex[k]` returns
the Set of k-cells; `boundary(k)` returns ∂_k: C_k → C_{k-1}; the identity
∂² = 0 earns the class. `Mesh(CellComplex)` adds a `chart: Chart` from
`continuous/`: faces are regions in the chart's parameter space (not polygons
in physical space); cell volumes are derived from face geometry via the
divergence theorem itself (`|Ωᵢ| = (1/n) ∑_{f ∈ ∂Ωᵢ} xf · nf Af`); areas and
volumes in non-Cartesian geometries use `√|g|` from the metric of the
chart's domain manifold. The restriction operator `Rₕ` (free: `mesh: Mesh`)
is the formal bridge: a continuous Function plus a Mesh yields a MeshFunction
via `(Rₕ f)ᵢ = |Ωᵢ|⁻¹ ∫_Ωᵢ f dV`; the output has `.mesh == Rₕ.mesh` by
construction. `StructuredMesh` adds `coordinate(idx)` and the derived
evaluation bridge `field.expr.subs(zip(chart.symbols, coordinate(idx)))`.

**Next steps.**

**`geometry/` package** (in progress). Concrete instantiable objects —
`CartesianMesh`, `SchwarzschildManifold`, and future common geometries —
belong in `geometry/`, not in the abstract layers. `CartesianMesh` and
`EuclideanManifold`/`CartesianChart` have been introduced; `RotatingChart`
(Epoch 7) and `SchwarzschildManifold` (moved from `validation/`) will
follow in later epochs.

**`RotatingChart` design.** The formally principled approach to rotating
reference frames is to change the metric, not add source terms. A
`RotatingChart` in `geometry/` carries the rotating-frame metric; fictitious
forces (centrifugal, Coriolis) appear as Christoffel symbols of that metric,
not as ad hoc source terms. This is co-designed with Epoch 7 hydro validation
tests so that the design is grounded by a concrete simulation before it is
declared stable.

---

## Physics roadmap

Each physics epoch adds new fields and equations to the continuous layer and
extends the discrete and numerical layers minimally to evaluate them.

### Foundation epochs

| Epoch | Layer | Capability |
|-------|-------|------------|
| 0 | — | Project scaffolding: CI, pre-commit, documentation standards. ✓ |
| 1 | Continuous | `continuous/` ABCs: full manifold and field hierarchy, operators, boundary conditions, metric; coordinate structure (`Chart`, `Atlas`). `foundation/` ABCs: `Set`, `Function`, `IndexedSet`, `IndexedFamily`. ✓ |
| 2 | Continuous + Discrete + Geometry | `ConservationLaw(DifferentialOperator)` in `continuous/` (divergence form, flux + source, integral form via divergence theorem). In `discrete/`: `CellComplex(IndexedFamily)` (chain (C_*, ∂), ∂²=0); `Mesh(CellComplex)` with `chart: Chart`; restriction operator `Rₕ`; `StructuredMesh(Mesh)` with `coordinate(idx)`, narrows `complex[n]` to `IndexedSet`; `MeshFunction` with `.mesh` accessor. In `geometry/`: `EuclideanManifold(RiemannianManifold)` (flat ℝⁿ, g = I), `CartesianChart` (identity chart), `CartesianMesh(StructuredMesh)` (derives all geometry from origin/spacing/shape). |
| 3 | Discrete | `Discretization(NumericFunction[ConservationLaw, DiscreteOperator])`: maps conservation law + mesh + order to a `DiscreteOperator` via commutation diagram `Lₕ ∘ Rₕ ≈ Rₕ ∘ L` at `O(hᵖ)`; `DiscreteOperator` earns `.mesh: Mesh` (same-mesh constraint). Truncation error verified algebraically via SymPy. First working Poisson solver on `CartesianMesh`. |
| 4 | Numerical | JAX evaluation layer: concrete field storage as `jax.Array`; JIT-compiled stencil application; explicit time integration; HDF5 I/O with provenance. |

### Physics epochs

| Epoch | Capability |
|-------|------------|
| 5 | Scalar transport: linear advection and diffusion on a `CartesianMesh` via FVM. First end-to-end simulation; validates the full pipeline. |
| 6 | Newtonian hydrodynamics: Euler equations, FVM Godunov, PPM reconstruction, HLLC/HLLE Riemann solvers. |
| 7 | Rotating reference frames: `RotatingChart` in `geometry/`; formally principled approach via metric change (fictitious forces = Christoffel symbols of the rotating-frame metric, not source terms); co-designed with Epoch 6 hydro validation tests. |
| 8 | Self-gravity: multigrid Poisson solver; particle infrastructure. |
| 9 | Microphysics: EOS interface, reaction networks, cooling tables, opacities. |
| 10 | MHD: ideal and resistive, constrained transport, super-time-stepping. |
| 11 | Radiation transport: gray FLD, multigroup FLD, two-moment M1. |
| 12 | AMR: adaptive mesh refinement hierarchy, coarse–fine interpolation, load balancing. |
| 13 | Special and general relativity: SR hydro, GR hydro/MHD on fixed spacetimes, dynamical spacetime via BSSN. |
| 14 | Particle cosmology: SPH, meshless methods, FRW integrator, halo finders. *(stretch)* |
| 15 | Moving mesh: Arepo-class Voronoi tessellation. *(stretch)* |
| 16 | Stellar evolution: 1-D Lagrangian solver with nuclear burning and mixing. *(stretch)* |
| 17 | Subgrid physics and synthetic observables: plugin interface, in-situ rendering. *(stretch)* |

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
