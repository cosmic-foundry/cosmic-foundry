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

### foundation/

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

### continuous/

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
└── ConservationLaw                          — divergence form: ∂ₜU + ∇·F(U) = S
                                               free: flux: Function[Field, TensorField], source: Field
                                               earned by: integral form ∮_∂Ωᵢ F·n dA = ∫_Ωᵢ S dV
                                               is fully determined by flux + divergence theorem;
                                               not derivable from bare DifferentialOperator

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

**`ConservationLaw` is spatial only.** `∂ₜ` is handled by the time integrator
(Epoch 2), not this object. This separation is preserved under the 3+1 ADM
decomposition: in GR, covariant equations `∇_μ F^μ = S` decompose to
`∂ₜ(√γ U) + ∂ᵢ(√γ Fⁱ) = √γ S(α, β, γᵢⱼ, Kᵢⱼ)` — still a spatial
divergence operator with metric factors entering through the `Chart` and
curvature terms in `source`. `ConservationLaw` is stable through Epoch 10.

**Planned additions** (Epoch 10)

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

### discrete/

```
CellComplex(IndexedFamily)     — chain (C_*, ∂): complex[k] → Set of k-cells;
                                  boundary operators ∂_k: C_k → C_{k-1};
                                  earned by ∂² = 0 (∂_{k-1} ∘ ∂_k = 0)
└── Mesh(CellComplex)          — adds chart: Chart; grounds the complex geometrically;
                                  cell volumes derived via divergence theorem:
                                    |Ωᵢ| = (1/n) ∑_{f ∈ ∂Ωᵢ} xf · nf Af
                                  general volumes/areas: ∫ √|g| dV and ∫ √|g_σ| dA;
                                  earned by: volume, area, normal are derived properties
                                  fully determined by CellComplex + Chart;
                                  covers Cartesian (g = I), cylindrical (√|g| = r),
                                  GR spacetimes (curved g), moving mesh (time-varying Chart)
    └── StructuredMesh(Mesh)   — abstract: coordinate(idx) → ℝⁿ;
                                  evaluation bridge:
                                    field.expr.subs(zip(chart.symbols, coordinate(idx)))
                                  narrows complex[n] from Set to IndexedSet: regularity
                                  implies top-dimensional cells biject with a rectangular
                                  region of ℤⁿ

MeshFunction(NumericFunction[Mesh, V])
                               — value assignment to mesh elements (cells, faces, vertices);
                                  earned by .mesh: Mesh typed accessor,
                                  by analogy with Field.manifold

RestrictionOperator(NumericFunction[Function[M,V], MeshFunction[V]])
                               — free: mesh: Mesh;
                                  (Rₕ f)ᵢ = |Ωᵢ|⁻¹ ∫_Ωᵢ f dV;
                                  formal bridge from continuous/ to discrete/:
                                  a Function plus a Mesh yields a MeshFunction;
                                  the restriction depends on both — neither alone suffices
```

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

**Planned additions (Epoch 1 — Discrete operators):**

```
Discretization(NumericFunction[ConservationLaw, DiscreteOperator])
                            — free: mesh: Mesh
                              maps a ConservationLaw to a DiscreteOperator;
                              encapsulates the scheme choice (reconstruction,
                              numerical flux, quadrature, boundary condition).
                              Defined by the commutation diagram:
                                Lₕ ∘ Rₕ ≈ Rₕ ∘ L   (up to O(hᵖ))
                              The approximation order p is a property of the
                              concrete scheme, proved by its convergence test —
                              not a parameter of the abstract interface.
                              The commutation check verified algebraically via
                              SymPy is the machine-checkable derivation required
                              by Lanes B and C.
                              Formally separate from Rₕ: Rₕ projects field values
                              (Function → MeshFunction); Discretization projects
                              operators (ConservationLaw → DiscreteOperator).
└── FVMDiscretization       — free: mesh, reconstruction, boundary_condition
                              concrete FVM scheme; generic over ConservationLaw.
                              For each cell Ωᵢ, evaluates ∮_∂Ωᵢ F·n̂ dA by
                              composing the conservation law's flux function with
                              a FaceReconstruction at each face of Ωᵢ; the BC
                              enters through boundary_condition (see below).
                              Not specialized to any particular conservation law:
                              the same class produces the Poisson operator in
                              Epoch 1 (flux = ∇φ, linear reconstruction) and the
                              Euler operator in Epoch 4 (flux = (ρv, ρv⊗v+pI,
                              (E+p)v), MUSCL/PPM reconstruction with Riemann
                              solver). Specializations belong in the flux
                              function and the FaceReconstruction — not in a
                              new Discretization subclass per equation.

DiscreteOperator(NumericFunction[MeshFunction, MeshFunction])
                            — the output of Discretization; the Lₕ that makes
                              Lₕ ∘ Rₕ ≈ Rₕ ∘ L hold to the chosen order.
                              Earns its class via .mesh: Mesh — constrains input and
                              output to the same mesh (operator.mesh == input.mesh ==
                              output.mesh), by analogy with DifferentialOperator.manifold.
                              Not independently constructed from stencil coefficients.

FaceReconstruction          — free: order p
                              reconstructs the values needed for face-flux
                              evaluation from cell averages on a stencil around
                              the face. The interface adapts per flux type:
                              for linear diffusive flux (Poisson), reconstructs
                              the face-centered gradient as a single value; for
                              nonlinear hyperbolic flux (Euler), produces a
                              two-sided state (U_L, U_R) that a Riemann solver
                              consumes. Machine-checkable derivation (Lane C):
                              symbolic Taylor expansion of the reconstruction
                              against the progenitor face value yields leading
                              error O(hᵖ) for smooth test functions.
                              Epoch 1 concrete classes:
                              CenteredDifferenceGradient(p=2),
                              CenteredPolynomialGradient(p=4).
                              Epoch 4 extends the family with MUSCL(p=2) and
                              PPM(p=4) for hydro; the ABC is designed for that
                              reuse from the outset.

LinearSolver(NumericFunction[MeshFunction, MeshFunction])
                            — given a DiscreteOperator Lₕ and rhs MeshFunction f,
                              solves Lₕ u = f. Interface is general enough for
                              matrix-free, sparse, and dense implementations; the
                              choice is a concrete-class concern, not encoded in
                              the abstract signature.
                              Epoch 1 ships DenseDirectSolver first: assembles
                              the dense matrix by applying Lₕ to unit MeshFunctions
                              and calls LAPACK. Easiest to interpret, exact up to
                              floating-point error, and the assembled matrix is a
                              ground-truth reference for the sparse/matrix-free
                              variants that follow. Iterative variants (CG,
                              multigrid) land in Epoch 6 (self-gravity).
```

**Boundary condition application (Option B, Epoch 1 decision).** `FVMDiscretization`
takes the `BoundaryCondition` as a constructor parameter; the resulting
`DiscreteOperator` is the discrete analog of `L` on the constrained function
space `{φ : Bφ = g}`. This keeps the commutation diagram a property of a single
operator, and lets the Epoch 6 multigrid ask the discretization for coarse
operators rather than asking the operator for its BC. Not committed long-term:
if time-dependent `g` arrives with Epoch 4 hydro (inflow/outflow BCs that change
per step), BC can migrate to a solver-level parameter without breaking the
interior-flux derivation — the interior `Lₕ` and the face-reconstruction family
are independent of where BC is injected.

### geometry/

```
EuclideanManifold(RiemannianManifold)  — flat ℝⁿ; metric g = δᵢⱼ; free: ndim, symbol_names

CartesianChart(Chart)                  — identity map φ: ℝⁿ → ℝⁿ on a EuclideanManifold;
                                         derived: inverse = self, symbols from domain

CartesianMesh(StructuredMesh)          — free: origin, spacing, shape;
                                         derived: chart = CartesianChart on EuclideanManifold(ndim)
                                                  coordinate = origin + (idx + ½)·spacing
                                                  cell volume = ∏ Δxₖ
                                                  face area = ∏_{k≠j} Δxₖ  (face ⊥ axis j)
                                                  face normal = ê_j
```

### computation/

JAX evaluation. The only layer that touches floats. Planned: concrete field
storage as `jax.Array`; JIT-compiled stencil application; explicit time
integration; HDF5 I/O with provenance.

### Cross-cutting

**Numerical transcription discipline.**
Physics capabilities sourced from reference tables (EOS polynomial fits,
reaction networks, opacity tables) need a discipline governing how
numeric tables are transcribed, verified, and updated independently of
the derivation-first lane policy. This decision is deferred to Epoch 7
(microphysics), when the first such capability lands.

**Kernel composition model.**
A backend-agnostic interface separating kernel computation (Op) from
spatial domain and execution policy (Policy) is a design goal. An
earlier Op/Policy/Dispatch framing was dropped before it was realized.
The formal model governing composition, backend substitutability, and
dispatch is unsettled.

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

**Epoch 1 Poisson sprint.** The target is a working FVM Poisson solver on
`CartesianMesh` with Dirichlet boundary conditions, verified against an
analytic solution. The sprint is structured as six PRs (C1–C6); each earns
its scope by a Lane C symbolic derivation and each introduces only objects
justified by a falsifiable constraint. The ambition is not "a working
Poisson solver" — it is the reusable FVM machinery the rest of the engine
is built on. Epoch 4 (hydro) swaps the `ConservationLaw` and the
`FaceReconstruction`; the `FVMDiscretization`, `LinearSolver`, and
`BoundaryCondition` machinery is unchanged.

**C1 — Continuous progenitors.** Add `GradientOperator(DifferentialOperator)`
and `LaplaceOperator = Div ∘ Grad` in `theory/continuous/`; add
`PoissonEquation(ConservationLaw)` with `flux = ∇φ`, `source = ρ`. Lane C:
symbolic verification that the Cartesian-coordinate expansion of `∇²` equals
`Σ_a ∂²/∂x_a²`, and that the divergence-theorem form of `PoissonEquation`
recovers the standard `∇²φ = ρ` under the Cartesian chart. No discrete code.

**C2 — Full chain complex on `CartesianMesh`.** Extend
`CartesianMesh.boundary(k)` to all k ∈ [1, n]; verify `∂_{k−1} ∘ ∂_k = 0`
symbolically in the `IndexedSet` of cells for n ∈ {1, 2, 3}. The face-sum
machinery used by `FVMDiscretization` to assemble `∮_∂Ωᵢ F·n̂ dA` reads the
signed incidence from `boundary(n)`; the lower-k operators are carried
because `CellComplex` earns its class by `∂² = 0` everywhere, not only at
the top dimension. Lane C.

**C3 — `FaceReconstruction` family (p = 2 and p = 4 together).**
Introduce the `FaceReconstruction` ABC and ship `CenteredDifferenceGradient`
(p = 2, three-cell stencil) *and* `CenteredPolynomialGradient` (p = 4,
five-cell stencil) in the same PR. Shipping both at once forces the ABC
to actually generalize rather than codify the p = 2 case. Lane C per
concrete class: symbolic Taylor expansion of the reconstructed face
gradient against the exact face gradient of a smooth test function yields
leading error O(hᵖ).

**C4 — Generic `FVMDiscretization`.** Introduce
`FVMDiscretization(mesh, reconstruction, boundary_condition)`; it is
generic over `ConservationLaw` — not Poisson-specific. The produced
`DiscreteOperator` computes `(Lₕ U)ᵢ = |Ωᵢ|⁻¹ Σ_f F(recon(U, f)) · n̂_f |f|`
where `F` comes from the conservation law's flux field and `recon` is
the `FaceReconstruction`. BC enters via the constructor parameter (see
"Boundary condition application" in `discrete/`). Lane C: verify the
commutation diagram `Lₕ Rₕ ≈ Rₕ L` at order p for `PoissonEquation`
paired with each of the two Epoch-1 reconstructions, symbolically.

**C5 — `LinearSolver` hierarchy with dense direct solver.** Introduce the
abstract `LinearSolver` interface and ship `DenseDirectSolver` as the first
concrete class: assembles the dense matrix by applying `Lₕ` to unit
`MeshFunction`s, then calls LAPACK. The assembled matrix is also the
ground-truth reference used to verify matrix-free and sparse variants
shipped later. Lane B: the solver reproduces, to floating-point precision,
a MeshFunction pre-computed from a known-conditioning test operator.

**C6 — End-to-end Poisson convergence test.** Compose `PoissonEquation`
(C1) + `CartesianMesh` with full chain complex (C2) + `FaceReconstruction`
(C3) + `FVMDiscretization` (C4) + Dirichlet `BoundaryCondition` +
`DenseDirectSolver` (C5) to solve `∇²φ = ρ` against the analytic solution
`φ = sin(πx)sin(πy)` on the unit square. Convergence test at N ∈ {8, 16,
32, 64}: O(h²) with `CenteredDifferenceGradient`, O(h⁴) with
`CenteredPolynomialGradient`. This is the externally-grounded verification
the epoch requires; the Lane C checks in C1–C4 are the derivation, C6
is the proof that the derivation was implemented.

---

## Physics roadmap

### Foundation epochs

| Epoch | Layer | Capability |
|-------|-------|------------|
| 1 | Discrete | **Discrete operators and first Poisson solver.** `Discretization` ABC + generic `FVMDiscretization(mesh, reconstruction, boundary_condition)` that is not specialized to any particular conservation law. `FaceReconstruction` family parameterized by order p (Epoch 1 concrete classes at p = 2 and p = 4, shipped together to force the ABC to generalize). `LinearSolver` hierarchy accommodating matrix-free, sparse, and dense implementations; `DenseDirectSolver` ships first. Boundary conditions enter through the discretization's constructor (option B). Truncation error verified algebraically via SymPy (commutation diagram `Lₕ Rₕ ≈ Rₕ L` at `O(hᵖ)`); convergence verified against `sin(πx)sin(πy)`. The FVM machinery produced here is the foundation for Epoch 4 hydrodynamics — swapping `ConservationLaw` and `FaceReconstruction` (MUSCL/PPM) reuses the same `FVMDiscretization`, `LinearSolver`, and BC pipeline. |
| 2 | Numerical | JAX evaluation layer: concrete field storage as `jax.Array`; JIT-compiled stencil application; explicit time integration; HDF5 I/O with provenance. |

### Physics epochs

| Epoch | Capability |
|-------|------------|
| 3 | Scalar transport: linear advection and diffusion on a `CartesianMesh` via FVM. First end-to-end simulation; validates the full pipeline. |
| 4 | Newtonian hydrodynamics: Euler equations, FVM Godunov, PPM reconstruction, HLLC/HLLE Riemann solvers. |
| 5 | Rotating reference frames: `RotatingChart` in `geometry/`; formally principled approach via metric change (fictitious forces = Christoffel symbols of the rotating-frame metric, not source terms); co-designed with Epoch 4 hydro validation tests. |
| 6 | Self-gravity: multigrid Poisson solver; particle infrastructure. |
| 7 | Microphysics: EOS interface, reaction networks, cooling tables, opacities. |
| 8 | MHD: ideal and resistive, constrained transport, super-time-stepping. |
| 9 | Radiation transport: gray FLD, multigroup FLD, two-moment M1. |
| 10 | AMR: adaptive mesh refinement hierarchy, coarse–fine interpolation, load balancing. |
| 11 | Special and general relativity: SR hydro, GR hydro/MHD on fixed spacetimes, dynamical spacetime via BSSN. |
| 12 | Particle cosmology: SPH, meshless methods, FRW integrator, halo finders. *(stretch)* |
| 13 | Moving mesh: Arepo-class Voronoi tessellation. *(stretch)* |
| 14 | Stellar evolution: 1-D Lagrangian solver with nuclear burning and mixing. *(stretch)* |
| 15 | Subgrid physics and synthetic observables: plugin interface, in-situ rendering. *(stretch)* |

### Per-epoch verification standard

Every epoch must satisfy this checklist before it is considered verified:

- Derivation document with SymPy checks for any new numerical scheme (Lanes B and C)
- At least one externally-grounded convergence test against an analytical solution
  or observational data (not an engine-generated golden file); where an analytical
  solution exists, the relevant `NumericFunction.symbolic` is declared so the
  check runs automatically
- Lane A/B/C classification stated in the PR description

---

## Platform milestones

| Milestone | Capability |
|-----------|------------|
| M0 | Process discipline: branch/PR/commit/attribution standards. ✓ |
| M1 | Verification infrastructure: convergence testing helpers, externally-grounded test pattern. ✓ |
| M2 | Documentation architecture: all live architectural decisions in `ARCHITECTURE.md`; `docs/` as API reference index. ✓ |
| M3 | Executable mathematical narrative: first `validation/` implementations (Schwarzschild spacetime, GPS time dilation); notebooks in `docs/` that import from `validation/` and run in CI. Settles coordinate-to-chart binding and the `SymbolicFunction` interface on concrete fields. ✓ |
| M4 | Validation infrastructure: manifests, provenance sidecars, comparison-result schema. Planned alongside Epoch 2. |
| M5 | Reproducibility capsule tooling: self-executing builder. |
| M6 | Application-repo capsule integration and multi-repository evidence regeneration. |
