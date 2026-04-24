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
├── GradientOperator                         — ∇: scalar Field → (0,1) TensorField;
│                                              earned by: derived order = 1
└── DivergenceFormEquation                   — ∇·F(U) = S in spatial-operator form;
                                               earned by: integral form ∮_∂Ωᵢ F·n dA = ∫_Ωᵢ S dV
                                               is fully determined by flux + divergence theorem,
                                               not derivable from bare DifferentialOperator.
                                               free: flux: Function[Field, TensorField], source: Field
                                               derived: order = 1
    ├── ConservationLaw                      — ∂ₜU + ∇·F(U) = S; F algebraic in U (hyperbolic).
    │                                          Example: Euler, F = (ρv, ρv⊗v + pI, (E+p)v).
    │                                          The ∂ₜ term is handled by the time integrator
    │                                          (Epoch 2), not by this object.
    │                                          Stable through Epoch 10.
    └── PoissonEquation                      — -∇²φ = ρ; earned by: derived flux = -∇(·).
                                               The sign convention (flux = -∇φ, not +∇φ) ensures
                                               the discrete operator is positive definite (see C4, C5).
                                               free: manifold, source; derived: flux = -∇(·), order = 1.
                                               There is no LaplaceOperator class: -∇²φ = -∇·∇φ is
                                               derivable from GradientOperator and the flux field.

Constraint(ABC)                       — interface: support → Manifold
└── BoundaryCondition                 — support is ∂M
    ├── LocalBoundaryCondition        — α·f + β·∂f/∂n = g; free: alpha, beta, constraint
                                        derived: support = constraint.manifold
    └── NonLocalBoundaryCondition     — constraint depends on values outside the immediate neighborhood
```

**`DivergenceFormEquation` subclass justification.** `ConservationLaw` earns
its class as the standard structure for hyperbolic systems (F algebraic in U);
it is a named mathematical concept in the literature, not a PDE-classification
marker. `PoissonEquation` earns its class by deriving `flux = -∇(·)`, removing
a free parameter from `DivergenceFormEquation`. Intermediate classification
ABCs (Elliptic, Parabolic, Hyperbolic as siblings) were considered and rejected:
none adds a derived property or type narrowing that mypy can check, so none
earns a class by the falsifiable-constraint rule.

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

**`DivergenceFormEquation` and its subtypes are spatial only.** `∂ₜ` is
handled by the time integrator (Epoch 2), not by these objects. For
`ConservationLaw` specifically, this separation is preserved under the 3+1
ADM decomposition: in GR, covariant equations `∇_μ F^μ = S` decompose to
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

**Discrete inner product.** Symmetry, positive-definiteness, and truncation
claims in this layer are stated in the cell-volume-weighted pairing
`⟨u, v⟩_h := Σᵢ |Ωᵢ| uᵢ vᵢ` — the ℓ²(h) analog of `∫_Ω uv dV`. This is
not a separate class (it carries no independent interface); it is a
conventional bilinear form used in proofs. The convergence norm on
`MeshFunction`s is the induced `‖u‖_{L²_h} := √⟨u, u⟩_h`; the local norm
for pointwise truncation claims is `‖u‖_{∞,h} := max_i |uᵢ|` over interior
cells.

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
                              interpreted on test fields f ∈ C^{p+2}(Ω); "≈"
                              means ‖Lₕ Rₕ f − Rₕ L f‖_{∞,h} = O(hᵖ) as h → 0,
                              measured in the local ℓ∞ norm over interior
                              cells. The approximation order p is a property
                              of the concrete scheme, proved by its
                              convergence test — not a parameter of the
                              abstract interface.
                              The commutation check verified algebraically via
                              SymPy is the machine-checkable derivation required
                              by Lanes B and C.
                              Formally separate from Rₕ: Rₕ projects field values
                              (Function → MeshFunction); Discretization projects
                              operators (ConservationLaw → DiscreteOperator).
└── FVMDiscretization       — free: mesh, numerical_flux, boundary_condition
                              concrete FVM scheme; generic over ConservationLaw.
                              For each cell Ωᵢ, evaluates ∮_∂Ωᵢ F·n̂ dA by
                              delegating to the NumericalFlux at each face; BC
                              enters through boundary_condition (see below).
                              Not specialized to any particular conservation law:
                              Epoch 1 supplies a DiffusiveFlux for Poisson;
                              Epoch 4 supplies a HyperbolicFlux for Euler.
                              Specializations belong in the NumericalFlux —
                              not in a new Discretization subclass per equation.
                              Note: LinearSolver is NOT part of the Epoch 4
                              reuse; the Euler equations are nonlinear and need
                              a separate NonlinearSolver / Newton iteration.

DiscreteOperator(NumericFunction[MeshFunction, MeshFunction])
                            — the output of Discretization; the Lₕ that makes
                              Lₕ ∘ Rₕ ≈ Rₕ ∘ L hold to the chosen order.
                              Earns its class via .mesh: Mesh — constrains input and
                              output to the same mesh (operator.mesh == input.mesh ==
                              output.mesh), by analogy with DifferentialOperator.manifold.
                              Not independently constructed from stencil coefficients.

NumericalFlux               — free: order: int
                              given cell averages U and a face, returns
                              F·n̂·|face_area|. order is the COMPOSITE
                              convergence order of the scheme:
                                order = min(reconstruction_order,
                                            face_quadrature_order,
                                            deconvolution_order)
                              Each of the three components is a distinct
                              operator with its own Lane C expansion:
                                • Reconstruction R_p: cell averages → polynomial
                                  representation; Taylor expansion in h shows
                                  leading error O(h^{p_R}) against the exact
                                  pointwise value.
                                • Face quadrature Q_p: integrates the polynomial
                                  flux over the face; midpoint (O(h²)) or
                                  Simpson (O(h⁴)) rule; Lane C: quadrature error
                                  against the exact face average of a smooth
                                  test function.
                                • Deconvolution D_p: corrects between cell-average
                                  and point-value representations,
                                    Uᵢ = Ū_i - (h²/24)(∇²U)ᵢ + O(h⁴)  (p=4)
                                    Uᵢ = Ū_i + O(h²)                  (p=2, identity)
                                  Lane C: Taylor expansion of the finite-average
                                  operator confirms the stated residual.
                              All three must be ≥ order; the class is
                              responsible for ensuring they are.
                              Earned by: order is a verifiable claim —
                              the Lane C Taylor expansion of the composite
                              face flux F_face = Q_p ∘ F ∘ D_p ∘ R_p against
                              the exact face-averaged flux of a smooth test
                              function yields leading error O(hᵖ),
                              where p = order.
├── DiffusiveFlux(order)    — free: order: int. F(U) = ∇U; constructs the
│                             appropriate stencil, face-quadrature rule,
│                             and cell-average/point-value deconvolution for
│                             that order. One class, not one class per order:
│                             DiffusiveFlux(2) and DiffusiveFlux(4) are
│                             *instances*, not subclasses. The test that forces
│                             generalization is that both instances pass the
│                             same Lane C contract.
└── HyperbolicFlux(order, riemann_solver)
                            — free: order: int, riemann_solver: RiemannSolver.
                              F(U) nonlinear; reconstruction at the given order
                              produces a two-sided state (U_L, U_R) that the
                              Riemann solver consumes. Epoch 4 ships
                              HyperbolicFlux(2, HLLC) and HyperbolicFlux(4, HLLC)
                              as instances — not subclasses.

LinearSolver                — solves Lₕ u = f for a *linear* DiscreteOperator Lₕ.
                              SCOPE: linear operators only. Epoch 4 hydro (nonlinear
                              flux) requires a separate NonlinearSolver / Newton
                              iteration. LinearSolver is not the shared machinery
                              for Epoch 4; only FVMDiscretization and NumericalFlux
                              are reused across epochs.
                              Epoch 1 ships DenseJacobiSolver: assembles the
                              dense (N^d × N^d) matrix on a d-dimensional grid
                              with N cells per axis, by applying Lₕ to unit
                              MeshFunctions ordered lexicographically
                              (idx → Σ_a idx[a]·N^a). It iterates Jacobi sweeps
                              until residual tolerance ‖f − Lₕ u‖_{L²_h} < τ.
                              All linear algebra hand-rolled — no LAPACK, no
                              external solvers. Jacobi convergence rate is
                              O(1/h²) iterations for the DiffusiveFlux(2)
                              Poisson operator; C6 convergence tests cap at
                              N ≤ 32 in 2-D (≤ 1024 unknowns) accordingly.
                              Performance optimization deferred.
```

**Boundary condition application (Option B, Epoch 1 decision).** `FVMDiscretization`
takes the `BoundaryCondition` as a constructor parameter; the resulting
`DiscreteOperator` is the discrete analog of `L` on the constrained function
space `{φ : Bφ = g}`. This keeps the commutation diagram a property of a single
operator, and lets the Epoch 6 multigrid ask the discretization for coarse
operators rather than asking the operator for its BC. Not committed long-term:
if time-dependent `g` arrives with Epoch 4 hydro (inflow/outflow BCs that change
per step), BC can migrate to a solver-level parameter without breaking the
interior-flux derivation — the interior `Lₕ` and the numerical-flux family
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
analytic solution. The sprint is structured as eight PRs (C1–C8); each earns
its scope by a Lane C symbolic derivation and each introduces only objects
justified by a falsifiable constraint. The ambition is not "a working
Poisson solver" — it is the reusable FVM machinery the rest of the engine
is built on. Epoch 4 (hydro) swaps the `ConservationLaw` and the
`NumericalFlux`; the `FVMDiscretization` and `BoundaryCondition` machinery
is unchanged. `LinearSolver` is NOT part of the Epoch 4 reuse: the Euler
equations are nonlinear and require a separate `NonlinearSolver`.

**C1 — Continuous progenitors. ✓** Added `GradientOperator(DifferentialOperator)`
(derived `order = 1`) and `DivergenceFormEquation(DifferentialOperator)` as the
parent for all divergence-form PDEs. `ConservationLaw` now subclasses
`DivergenceFormEquation`. `PoissonEquation(DivergenceFormEquation)` is an ABC
with `flux = -∇(·)` derived and `manifold`/`source` abstract; it earns its
class by fixing the flux, removing a degree of freedom from
`DivergenceFormEquation`. `Discretization` type parameter updated from
`ConservationLaw` to `DivergenceFormEquation`. Intermediate classification ABCs
(Elliptic, Parabolic, Hyperbolic as siblings) were not introduced: none earns a
class by the falsifiable-constraint rule. Lane C verified: `∇·(-∇φ) = -∇²φ = ρ`
symbolically in `tests/test_poisson_equation.py`.

**C2 — Full chain complex on `CartesianMesh`.** Extend
`CartesianMesh.boundary(k)` to all k ∈ [1, n]; verify `∂_{k−1} ∘ ∂_k = 0`
symbolically in the `IndexedSet` of cells for n ∈ {1, 2, 3}. The face-sum
machinery used by `FVMDiscretization` to assemble `∮_∂Ωᵢ F·n̂ dA` reads the
signed incidence from `boundary(n)`; the lower-k operators are carried
because `CellComplex` earns its class by `∂² = 0` everywhere, not only at
the top dimension. Lane C.

**Open question before C2 can open.** A 3-D Cartesian grid has three disjoint
`IndexedSet`s of faces (one per axis orientation). The existing
`CellComplex.complex[k] → Set` signature has not been examined for whether
`Set` can represent this disjoint union, or whether a richer return type is
needed for k < n. This data-structure question must be answered and the
decision recorded in ARCHITECTURE.md before C2 is opened.

**C3 — `NumericalFlux` family (order = 2 and order = 4 together).**
Introduce the `NumericalFlux` ABC and the `DiffusiveFlux(order)` concrete
class. Construct `DiffusiveFlux(2)` *and* `DiffusiveFlux(4)` — two
instances of the same class — and verify both in the same PR. The test that
forces generalization is not "two subclasses pass the same test" but "one
class parameterized by `order` satisfies the same Lane C contract at both
orders." Shipping an `order` parameter that only changes the stencil width
would fail: the ORDER of a FVM scheme is
`min(reconstruction_order, face_quadrature_order, deconvolution_order)`.
`DiffusiveFlux(order)` must independently configure all three components
for each `order`. Lane C per instance requires eight separate symbolic
checks: Taylor expansion of reconstruction, face-quadrature, deconvolution,
and composite face flux — each against the exact face-averaged flux — for
both p=2 and p=4. Each component must independently achieve the stated order,
and the composite (their composition) must yield leading error O(hᵖ) where
p = order. The `NumericalFlux` ABC defines `free: order: int`; concrete
subclasses may introduce additional constructor parameters specific to the
flux family (e.g. `HyperbolicFlux(order, riemann_solver)` adds a Riemann
solver, while `DiffusiveFlux(order)` does not).

**C4 — Generic `FVMDiscretization` with commutation Lane C.** Introduce
`FVMDiscretization(mesh, numerical_flux, boundary_condition)`; it is
generic over `ConservationLaw` — not Poisson-specific. The produced
`DiscreteOperator` computes `(Lₕ U)ᵢ = |Ωᵢ|⁻¹ Σ_f NF(U, f)` where `NF`
is the `NumericalFlux` evaluated at each face of Ωᵢ, with the conservation
law's flux function baked into `NF`. BC enters via the constructor parameter
(see "Boundary condition application" in `discrete/`). Lane C: verify the
commutation diagram `‖Lₕ Rₕ f − Rₕ L f‖_{∞,h} = O(hᵖ)` at order p for
`PoissonEquation` paired with `DiffusiveFlux(2)` and `DiffusiveFlux(4)`,
symbolically on test fields in `C^{p+2}(Ω)`. The SPD derivation is deferred
to C5.

**C5 — SPD analysis of the discrete Poisson operator.** For
`FVMDiscretization(PoissonEquation, DiffusiveFlux(order), DirichletBC)`
on `CartesianMesh`, the assembled operator is symmetric positive definite
with respect to the discrete inner product `⟨u, v⟩_h`. The chain:

1. *Symmetry* follows from the centered flux stencil and uniform cell
   volumes. Applying summation-by-parts to `⟨u, Lₕ v⟩_h`,
   `Σᵢ |Ωᵢ| uᵢ (Lₕ v)ᵢ = Σ_faces (area/h_⊥)·(u_+ − u_−)(v_+ − v_−)`,
   which is manifestly symmetric in `(u, v)`. The identity holds for any
   centered `DiffusiveFlux(order)` at every interior face.
2. *Positive definiteness* follows from the sign convention. With
   `flux = -∇φ`, Lₕ is the discrete analog of `-∇²`. Setting `u = v` in
   (1) yields `⟨u, Lₕ u⟩_h = Σ_faces (area/h_⊥)·(u_+ − u_−)² ≥ 0`.
   Equality forces `u_+ = u_−` across every interior face; together with
   `u_boundary = 0` from Dirichlet BC, this forces `u ≡ 0`. Hence
   `⟨u, Lₕ u⟩_h > 0` for all `u ≠ 0`.
3. *Spectral inheritance.* Step 2 is the discrete analog of L² positive-
   definiteness of `-∇²`. The explicit eigenvalues quoted in C6 are a
   consequence of SPD + translation invariance on `CartesianMesh`, not
   additional hypotheses.

The row ordering for matrix assembly is lexicographic
(idx → Σ_a idx[a]·N^a); unit-basis assembly `A eⱼ = Lₕ eⱼ` fills one
column per cell. Lane C verifies SPD symbolically at N = 4 in 1-D and
2-D for both `DiffusiveFlux(2)` and `DiffusiveFlux(4)`, so the assertion
does not depend on a numerical eigenvalue computation.

**C6 — `LinearSolver` hierarchy with `DenseJacobiSolver`.** Introduce the
abstract `LinearSolver` interface, scoped explicitly to *linear* operators
(nonlinear problems need a separate `NonlinearSolver`). This PR develops
the interface, the `DenseJacobiSolver` implementation with matrix assembly
via unit basis, and the Jacobi spectral-radius derivation for `DiffusiveFlux(2)`.
The convergence-count Lane B check for order=4 is deferred to C7. The
derivation works simultaneously in two directions. Both directions are stated
for `FVMDiscretization(PoissonEquation, DiffusiveFlux(2), DirichletBC)` on
`CartesianMesh`; the same construction applies to `DiffusiveFlux(4)` but
the explicit spectral rate is different — see the "Order ≥ 4" remark below.

*Forward from the formal ingredients already in the code.* At the point C6
runs, three objects are in hand:
1. The `DiscreteOperator` Lₕ. *Linearity of Lₕ is specific to this
   specialization*: `DiffusiveFlux` produces a centered-difference stencil
   that is an affine combination of cell values, so the induced operator
   is linear. For `HyperbolicFlux` (Epoch 4) Lₕ is nonlinear and this
   derivation does not apply — hence `LinearSolver` is scoped away from
   the Euler path in Epoch 4.
2. The assembled dense `(N^d × N^d)` matrix `A`, obtained by applying Lₕ
   to each unit-basis `MeshFunction` in lexicographic order (one column
   per cell).
3. The SPD property of A, proved in C5's Lane C derivation (not asserted
   here) from summation-by-parts plus the sign convention `flux = -∇φ`.

From SPD alone, the equation `Lₕ u = f` is equivalent to
`u = u + α(f − Au)` for any scalar α — every solution is a fixed point
of this map. The map is a contraction iff `ρ(I − αA) < 1`, guaranteed
for α ∈ (0, 2/λ_max) by SPD. Preconditioning by an easily invertible
approximation to A accelerates convergence; the diagonal `D = diag(A)`
is the simplest such choice.

*D is invertible* by a weak-diagonal-dominance + irreducibility argument,
not strict dominance. The constrained operator on `{φ : φ|∂Ω = g}` is
equivalent, after eliminating boundary unknowns via affine substitution,
to the interior operator on `{φ_interior}` with modified RHS; diagonal
dominance is evaluated on this reduced operator. Interior rows of the
reduced system satisfy `A_{ii} = Σ_{j≠i} |A_{ij}|` (equality, weak);
the reduction to interior-only unknowns automatically ensures all remaining
rows have strict diagonal dominance (because one stencil neighbor per
boundary-adjacent cell is absorbed into the RHS by Dirichlet elimination).
The mesh-cell adjacency graph is connected — a fact earned by `CellComplex`
being irreducible in the sense that every cell reaches every other via
repeated applications of `boundary(n)`. Weak dominance everywhere + strict
dominance somewhere + irreducibility is the hypothesis of the Taussky
theorem: A is invertible, and every diagonal entry is strictly positive
(so D⁻¹ exists). The resulting fixed-point map `u^{k+1} = D⁻¹(f − (A − D)u^k)`
is Jacobi — arrived at from the ingredients, not imported as a recipe.

*Backward from known convergence properties.* For `DiffusiveFlux(2)` the
eigenstructure of Lₕ on `CartesianMesh` with Dirichlet BC is computable
in closed form. In the discrete inner product `⟨·,·⟩_h` the eigenvalues
are `λ_k = (2/h²) Σ_a (1 − cos(kₐπh))` for multi-indices
`k ∈ {1, …, N−1}^d` — the discrete analog of the continuous Laplacian
spectrum `π²|k|²`, recovering it exactly as `h → 0`. With diagonal
entries `D_{ii} = 2d/h²`, the Jacobi iteration matrix `M_J = D⁻¹(A − D)`
has eigenvalues `μ_k = (1/d) Σ_a cos(kₐπh)`, and spectral radius
`ρ(M_J) = cos(πh) = 1 − π²h²/2 + O(h⁴)` (attained at the smoothest mode
`k = (1,…,1)`). This is strictly less than 1, confirming convergence;
iterations to reduce residual by factor ε:
`⌈log ε / log cos(πh)⌉ ≈ 2 log(1/ε) / (π²h²)` — O(1/h²), derived from
the spectral bound, not asserted. The eigenvalue formula ties the solver
directly back to the continuous progenitor `-∇²`; the convergence
guarantee comes from the same spectral theory that the commutation
diagram verifies.

*Order ≥ 4 remark.* For `DiffusiveFlux(4)` the closed-form eigenvalues
above do not apply; the wider stencil introduces different Fourier
symbols. SPD (from C5) still guarantees convergence qualitatively for
any α small enough, but the iteration-count bound must be re-derived
numerically by a one-off dense eigenvalue scan on a representative
grid. The empirical rate for `DiffusiveFlux(4)` is deferred to C7; the
closed-form spectral derivation is deferred and re-opened when multigrid
(Epoch 6) requires spectral bounds on wide-stencil operators.

All linear algebra is hand-rolled — no NumPy `linalg`, no LAPACK.

**C7 — DenseJacobiSolver convergence check (order=4 Lane B).** Verify that
`DenseJacobiSolver` reaches the prescribed tolerance within tractable
iteration counts on representative grids for `DiffusiveFlux(4)`. The O(1/h²)
iteration count bounds C8 to N ≤ 32 in 2-D. Lane B: on an N = 8 system,
verify that the solver reaches the prescribed tolerance within the
iteration count implied by the SPD property and empirical spectral radius,
and that the residual `‖f − Lₕ u^k‖_{L²_h}` decreases monotonically.

**C8 — End-to-end Poisson convergence test.** Compose `PoissonEquation`
(C1) + `CartesianMesh` with full chain complex (C2) + `DiffusiveFlux(2)`
and `DiffusiveFlux(4)` (C3) + `FVMDiscretization` (C4) + SPD analysis (C5) +
Dirichlet `BoundaryCondition` + `DenseJacobiSolver` (C6, C7) to solve
`-∇²φ = ρ` against the analytic solution `φ = sin(πx)sin(πy)` on the
unit square. Convergence tests: N ∈ {8, 12, 16, 24, 32} (five points) for
p = 2; N ∈ {4, 6, 8, 12, 16} (five points) for p = 4 — capped to stay
above the h⁴ floating-point floor. The reported error is the cell-volume-
weighted discrete L² norm `‖φ_h − Rₕ φ_exact‖_{L²_h} = (Σᵢ |Ωᵢ|·(φ_h,ᵢ − (Rₕ φ)ᵢ)²)^{1/2}` —
the natural norm for the FVM formulation and the one in which the SPD
argument of C5 lives. A parallel max-norm `‖φ_h − Rₕ φ_exact‖_{∞,h}`
is also reported to detect pointwise failure modes (boundary-adjacent rows,
corners). The Lane C checks in C1–C5 are the derivation; C8 is the proof
that the derivation was implemented. C8 lives as a narrative application
in `validation/poisson/` with a mirror documentation page at `docs/poisson/`;
see the layout below.

**`validation/poisson/` and the Sphinx page.** C6 is a narrative
application, not only a test. It walks the pipeline from manufactured
solution to converged numerical result with every intermediate object
visible — mirroring the `validation/schwarzschild/` pattern (`# %%`
cells in a runnable Python file, plus a MyST page that re-executes in
the Sphinx build via `myst_nb`).

```
validation/poisson/
├── __init__.py
├── manufactured.py         — φ, ρ as SymbolicFunctions on EuclideanManifold(2):
│                               φ(x, y) = sin(πx) sin(πy)
│                               ρ(x, y) = -∇²φ = 2π² sin(πx) sin(πy)
│                             the identity -∇²φ − ρ = 0 is NOT checked at
│                             module load (import side-effects are avoided);
│                             it is verified in test_poisson_square.py.
├── poisson_square.py       — narrative script with `# %%` cells: compose
│                             PoissonEquation + CartesianMesh +
│                             DiffusiveFlux(order) + Dirichlet BC +
│                             FVMDiscretization + DenseJacobiSolver, solve,
│                             emit solution and convergence figures.
├── figures.py              — matplotlib figure functions (pure; returning Figure).
└── test_poisson_square.py  — machine-checked claims (pytest).
```

**Tests (`test_poisson_square.py`).** Seven claims, each independently
falsifiable:

1. *Manufactured pair identity.* Verify symbolically that `-∇²φ − ρ = 0`
   for the `manufactured` pair — not at module load, but here as a test.
2. *Commutation symbolic check on the test problem.* Using the
   `manufactured` pair, verify via SymPy that `Lₕ Rₕ φ − Rₕ Lφ` expanded
   at an interior cell has leading term `O(hᵖ)` for each `DiffusiveFlux(order)` instance.
   The derivation performed abstractly in C4 is re-executed on a concrete
   problem, catching any specialization bug.
3. *Numerical convergence, p = 2.* `assert_convergence_order(err_p2,
   [8, 12, 16, 24, 32], expected=2.0)` using the existing helper in
   `tests/utils/convergence.py`; the error is the cell-volume-weighted
   `L²_h` norm against `Rₕ manufactured.phi`.
4. *Numerical convergence, p = 4.* Same with `expected=4.0`; resolutions
   `[4, 6, 8, 12, 16]` — five points, capped below the h⁴ FP floor.
5. *Symmetry preservation.* `sin(πx)sin(πy)` is symmetric under `x ↔ y`;
   the numerical solution must respect this to floating-point precision
   for any N. A break signals a stencil-assembly bug.
6. *Operator symmetry and positive-definiteness.* For the assembled `Lₕ`
   matrix from C5, verify `⟨u, Lₕ v⟩_h = ⟨Lₕ u, v⟩_h` (symmetry) and
   `⟨u, Lₕ u⟩_h > 0` for `u ≠ 0` (positive-definiteness) on several
   random unit MeshFunctions `u, v`. Hand-rolled — no `np.linalg.cholesky`.
7. *Restriction commutes with boundary condition (nonzero data).* Using
   a separate test field `φ_bc(x, y) = x + y` (nonzero on all four sides),
   verify that `Rₕ φ_bc` on each boundary face matches the Dirichlet data
   analytically. The `sin(πx)sin(πy)` manufactured pair vanishes on `∂Ω`
   and cannot test this claim.

**Figures (`figures.py`).** Four pure functions, each returning a
`matplotlib.figure.Figure`:

- `solution_heatmap(N, p)` — `φ_numerical` as `imshow`, viridis, colorbar.
- `error_heatmap(N, p)` — signed `φ_numerical − φ_exact`, diverging
  colormap symmetric about 0; reveals whether the error is
  boundary-dominated or interior-dominated.
- `matrix_structure(N, p)` — `plt.spy(Lₕ)` at small N = 8, revealing the
  stencil pattern. Exact stencil width is determined in C3; do not
  presuppose it here.
- `convergence_figure()` — the headline figure: log-log max-norm error
  vs. `h` for both reconstructions, with reference lines at slopes 2 and
  4 and the measured slopes annotated.

**Documentation page (`docs/poisson/poisson_square.md`).** MyST notebook
re-executed at Sphinx build time. Structure chosen so the derivation is
visible in the rendered page, not only in test output:

1. *Problem statement.* `-∇²φ = ρ` on the unit square with Dirichlet BC;
   one code cell renders `sympy.Eq(lhs, rhs)` for the manufactured pair,
   so the symbolic identity is visible on the page.
2. *Continuous objects.* Instantiate `PoissonEquation(flux, source)`;
   display `flux.expr` and `source.expr` to anchor the page in the C1
   progenitors.
3. *Mesh and chain complex.* Instantiate `CartesianMesh`; render the
   face-incidence list from `mesh.boundary(n)` as a small table at
   N = 4 — direct reuse of C2.
4. *NumericalFlux family.* Side-by-side table of stencil coefficients for
   `DiffusiveFlux(2)` and `DiffusiveFlux(4)`, derived symbolically. The page
   is about one class parameterized by `order`, not two separate classes.
5. *Discretization assembly.* `FVMDiscretization(mesh, numerical_flux, bc)`
   produces `Lₕ`; `matrix_structure` at N = 8 shown side-by-side for both
   instances.
6. *Solve.* `DenseJacobiSolver` applied; `solution_heatmap` and
   `error_heatmap` at N = 16 for each flux class (capped for build time).
7. *Convergence.* `convergence_figure()` inline; measured slopes
   annotated and compared to the expected 2 and 4.
8. *Derivation re-execution.* The symbolic `Lₕ Rₕ φ − Rₕ Lφ` expansion
   displayed as SymPy output — the truncation-error claim proved *in the
   rendered page*, not only in a test file.

The page is wired into `docs/index.md` under the "Validation" toctree as
`poisson/index`, next to `schwarzschild/index`. A one-line
`docs/poisson/index.md` with a toctree entry for `poisson_square`
matches the Schwarzschild pattern and keeps room for future
Poisson-family pages (Neumann, variable coefficient, 3-D).

****Docs/test code parity.** The documentation page runs the exact same code
as `test_poisson_square.py` — no specialized paths, no mocked data. The
Sphinx build may be slow as a result; static figure embedding is a deferred
optimization. A general mechanism for running only a cheaper subset in the
docs build (e.g. an environment variable honored by every validation
module's `resolutions` default) is worth considering as a shared pattern
across all validation pages, but not introduced as one-off code here.

**Open questions — Cross-epoch design points (Epoch 1 expected adaptation).** The Epoch 1 Poisson machinery lays the foundation for later physics epochs. Two adaptation points are expected to be designed in their respective epochs:

1. **AMR (Epoch 10).** `FVMDiscretization(mesh, numerical_flux, boundary_condition)`
   currently takes a fixed `Mesh`. AMR hierarchies (Epoch 10) will require
   localized discretization and coarse-grid operators across mesh levels.
   The `Discretization` interface and `DiscreteOperator` design are expected
   to generalize to hierarchical meshes; the specific adaptation (hierarchical
   discretization, prolongation/restriction operators, multigrid composition)
   is deferred to Epoch 10.

2. **GR (Epoch 11).** `NumericalFlux.__call__(U, face)` receives cell-average
   state and a face from a fixed mesh. In general relativity (Epoch 11) the
   face geometry is state-dependent: the 3-metric `γ_ij` is a dynamical field
   in the conservation law (via 3+1 ADM decomposition), so face areas and
   normals depend on the solution. The adaptation — passing metric-field
   state or chart information to the flux evaluator — is deferred to Epoch 11
   when `DynamicManifold` and time-evolved metrics are introduced.

---

## Physics roadmap

### Foundation epochs

| Epoch | Layer | Capability |
|-------|-------|------------|
| 1 | Discrete | **Discrete operators and first Poisson solver.** `DivergenceFormEquation` hierarchy in `continuous/` (`EllipticEquation`, `ParabolicEquation`, `ConservationLaw`). `Discretization` ABC + generic `FVMDiscretization(mesh, numerical_flux, boundary_condition)`. `NumericalFlux` family (`DiffusiveFlux` for Epoch 1; `HyperbolicFlux` for Epoch 4); order = min(reconstruction, face-quadrature, deconvolution) — all three independently verified. `LinearSolver` with `DenseJacobiSolver` (hand-rolled, dense, no LAPACK); scoped to linear operators only. Boundary conditions via discretization constructor. Truncation error proved symbolically; convergence verified against `sin(πx)sin(πy)`. FVM machinery reused in Epoch 4 by swapping `ConservationLaw` and `NumericalFlux`; `LinearSolver` is *not* reused (Euler is nonlinear). |
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
