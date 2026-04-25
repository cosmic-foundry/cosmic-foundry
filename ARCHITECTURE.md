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

DifferentialOperator(Function[Field, _C]) — L: Field → _C; interface: manifold, order
└── DivergenceFormEquation                   — ∇·F(U) = S in spatial-operator form;
                                               earned by: integral form ∮_∂Ωᵢ F·n dA = ∫_Ωᵢ S dV
                                               is fully determined by flux + divergence theorem,
                                               not derivable from bare DifferentialOperator.
                                               free: flux: Function[Field, TensorField], source: Field
                                               derived: order = 1
    └── PoissonEquation                      — -∇²φ = ρ; earned by: derived flux = -∇(·).
                                               The sign convention (flux = -∇φ, not +∇φ) ensures
                                               the discrete operator is positive definite (see C4, C6).
                                               free: manifold, source; derived: flux = -∇(·), order = 1.
                                               There is no LaplaceOperator class: -∇²φ = -∇·∇φ is
                                               the divergence of the flux field -∇φ; fully
                                               captured by the flux + divergence theorem.

Constraint(ABC)                       — interface: support → Manifold
└── BoundaryCondition                 — support is ∂M
    ├── LocalBoundaryCondition        — α·f + β·∂f/∂n = g; free: alpha, beta, constraint
                                        derived: support = constraint.manifold
    └── NonLocalBoundaryCondition     — constraint depends on values outside the immediate neighborhood
```

**`DivergenceFormEquation` subclass justification.** `PoissonEquation` earns
its class by deriving `flux = -∇(·)`, removing a free parameter from
`DivergenceFormEquation`. Classification ABCs (Elliptic, Parabolic, Hyperbolic,
ConservationLaw) were considered and rejected: none adds a derived property or
type narrowing that mypy can check — "F algebraic in U" and positivity of the
principal symbol are runtime mathematical properties, not structural constraints
expressible in the type hierarchy. None earns a class by the
falsifiable-constraint rule.

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
handled by the time integrator (Epoch 2), not by these objects. This separation
is preserved under the 3+1 ADM decomposition: in GR, covariant equations
`∇_μ F^μ = S` decompose to `∂ₜ(√γ U) + ∂ᵢ(√γ Fⁱ) = √γ S(α, β, γᵢⱼ, Kᵢⱼ)`
— still a spatial divergence operator with metric factors entering through the
`Chart` and curvature terms in `source`.

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
Discretization(NumericFunction[DivergenceFormEquation, DiscreteOperator])
                            — free: mesh: Mesh
                              maps a DivergenceFormEquation to a DiscreteOperator;
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
                              operators (DivergenceFormEquation → DiscreteOperator).
└── FVMDiscretization       — free: mesh, numerical_flux, boundary_condition
                              concrete FVM scheme; generic over DivergenceFormEquation.
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
                              Earns its class via two falsifiable claims:
                                order: int — composite convergence order
                                continuous_operator: DifferentialOperator —
                                  the continuous operator this approximates
                                  (added in C4; threaded automatically by
                                  Discretization from its input L)
                              Not independently constructed from stencil
                              coefficients; produced by a Discretization.

NumericalFlux(DiscreteOperator)
                            — a DiscreteOperator with the cell-average →
                              face-flux calling convention:
                                __call__(U: MeshFunction) → MeshFunction
                              where the returned MeshFunction is callable as
                              result((axis, idx_low)) and returns the flux
                              F·n̂·|face_area| at that face.  Inherits order
                              and (in C4) continuous_operator from
                              DiscreteOperator.  Full-field evaluation: all
                              face fluxes are available from one call; values
                              computed lazily on demand.
├── DiffusiveFlux(order)    — free: order: int. F(U) = -∇U; derives stencil
│                             coefficients symbolically in __init__ from the
│                             antisymmetric cell-average moment system.
│                             Validity: min_order=2, order_step=2 (even orders
│                             only; antisymmetric design kills odd error terms,
│                             constraining achievable orders to even integers).
│                             One class, not one class per order:
│                             DiffusiveFlux(2) and DiffusiveFlux(4) are
│                             *instances*, not subclasses.
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
                              Poisson operator; C9 convergence tests cap at
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
analytic solution. The sprint is structured as nine PRs (C1–C9); each earns
its scope by a Lane C symbolic derivation and each introduces only objects
justified by a falsifiable constraint. The ambition is not "a working
Poisson solver" — it is the reusable FVM machinery the rest of the engine
is built on. Epoch 4 (hydro) supplies a concrete `DivergenceFormEquation` for the Euler
equations and swaps the `NumericalFlux`; the `FVMDiscretization` and
`BoundaryCondition` machinery is unchanged. `LinearSolver` is NOT part of the Epoch 4 reuse: the Euler
equations are nonlinear and require a separate `NonlinearSolver`.

**C1 — Continuous progenitors. ✓** Added `DivergenceFormEquation(DifferentialOperator)`
as the parent for all divergence-form PDEs. `PoissonEquation(DivergenceFormEquation)`
is an ABC with `flux = -∇(·)` derived and `manifold`/`source` abstract; it
earns its class by fixing the flux, removing a degree of freedom from
`DivergenceFormEquation`. Classification ABCs (Elliptic, Parabolic, Hyperbolic,
ConservationLaw) and named operator ABCs (`GradientOperator`) were not
introduced: none earns a class by the falsifiable-constraint rule — the
identifying constraints (principal symbol structure, form degree) are beyond
Python's type system and are deferred to the form-degree redesign (see pre-C2
open question). Lane C verified: `∇·(-∇φ) = -∇²φ = ρ` symbolically in
`tests/test_poisson_equation.py`.

**C2 — Full chain complex on `CartesianMesh`. ✓** Extend
`CartesianMesh.boundary(k)` to all k ∈ [1, n]; verify `∂_{k−1} ∘ ∂_k = 0`
symbolically in the `IndexedSet` of cells for n ∈ {1, 2, 3}. The face-sum
machinery used by `FVMDiscretization` to assemble `∮_∂Ωᵢ F·n̂ dA` reads the
signed incidence from `boundary(n)`; the lower-k operators are carried
because `CellComplex` earns its class by `∂² = 0` everywhere, not only at
the top dimension. Lane C.

*Data structure decision (resolved at C2 open).* `Set` needs no change — it
is a pure marker with no abstract methods; concrete implementations carry
whatever indexing scheme they require.  k-cells in `CartesianMesh` are
identified by `(active_axes: tuple[int,...], idx: tuple[int,...])` where
`active_axes` is the sorted tuple of axes the cell extends along and `idx` is
the lower-corner position in the full vertex grid (`shape[a]` values along
active axes, `shape[a]+1` along inactive axes).  The boundary formula is the
standard CW orientation: for the j-th active axis aⱼ, the high face carries
sign (−1)ʲ and the low face carries sign (−1)ʲ⁺¹.  This resolves the
disjoint-family question: a face with `active_axes=(0,2)` and an edge with
`active_axes=(1,)` in the same mesh are distinguished by their axis sets, no
extra disjoint-union wrapper is needed.

*Pre-C2 open questions.*

*Phantom vs. real type parameters.* The general question underlying several
open decisions: for each generic type parameter in the hierarchy, is it **real**
(constrains something mypy checks) or **phantom** (documents mathematical intent
but provides no enforcement)?

Currently `D` and `C` in `Field[D, C]` are both phantom.  `C` is phantom
because `SymbolicFunction.__call__` overrides the generic return type to
`sympy.Expr` regardless of what `C` is bound to.  `D` is phantom because
`__call__` takes `*args: Any` — there is no Python type for "a point on a
manifold."  A manifold point has no representation in the type system: it is
only accessible as a coordinate tuple after choosing a chart.  `Chart[D, C]`
has the same problem: `__call__` returns coordinates (`tuple[Any, ...]`), not
an instance of the codomain type `C`.

Three decisions must be made before the type hierarchy can be considered
well-founded:

1. **The point type. ✓ Resolved.**  `D` in `Field[D, C]` is the manifold type
   (e.g. `EuclideanManifold`), not a coordinate-tuple type.  The distinction
   matters: a field's domain is the manifold M as a whole; the evaluation input
   is a point on M expressed in some chart.  These are different concepts.
   `Field.manifold -> D` returns the manifold object (domain in the
   field-theoretic sense); `field(point: Point[D]) -> C` evaluates via
   `__call__`, which accepts a typed `Point[D]` carrying the manifold, chart,
   and coordinates.  `Point[M]` is a frozen dataclass in
   `theory/continuous/manifold.py` (co-located with Chart to avoid a circular
   import) with fields `manifold: M`,
   `chart: Chart[M, Any]`, and `coords: tuple[Any, ...]`.  The chart is
   required so that evaluation can verify `point.chart.symbols == field.symbols`
   and raise `ValueError` on mismatch — catching cross-chart evaluation at
   runtime, and cross-manifold evaluation at the mypy level (a
   `Point[SchwarzschildManifold]` is rejected by mypy when passed to a
   `Field[EuclideanManifold, ...]`).  `SymbolicFunction` was moved from
   `theory/foundation/` to `theory/continuous/` (it depends on `Point` and
   `Chart`) and its `__call__` was tightened from `*args: Any` to
   `point: Point[M]`; `Function.__call__` at the foundation layer remains the
   generic `x: D` interface.

2. **Form-degree value types. ✓**  `ZeroForm[D]`, `OneForm[D]`, `TwoForm[D]`
   are named ABCs in `theory/continuous/differential_form.py`, each deriving
   `degree` from `DifferentialForm[D, C]` and fixing the Python value type:
   `C = sympy.Expr` for scalars, `C = tuple[sympy.Expr, ...]` for covectors,
   `C = sympy.Matrix` for antisymmetric rank-2 tensors.  `DifferentialOperator`
   is reparameterized from `Function[Field, _C]` to `Function[_D, _C]` with
   both TypeVars bound to `DifferentialForm`.

3. **`DivergenceFormEquation` consequent. ✓**  `DivergenceFormEquation` is
   `DifferentialOperator[DifferentialForm, ZeroForm]` — domain is open
   (`DifferentialForm`, any degree) pending multi-component input in Epoch 3;
   codomain is `ZeroForm` because ∇·F is always a scalar.  `flux` tightens to
   `Function[DifferentialForm, OneForm]` (the Riemannian metric isomorphism
   lets the flux live in Ω¹ rather than a general TensorField).  `source`
   tightens to `ZeroForm`.  `_NegatedGradientField` in `PoissonEquation`
   becomes a concrete `OneForm`; `_ZeroFormField` becomes a concrete `ZeroForm`.
   The `_D` domain sub-question (scalar vs. multi-component) is deferred to C3
   (Euler equations).

**C3 — `NumericalFlux` family (order = 2 and order = 4 together). ✓**
Introduced `NumericalFlux(DiscreteOperator)` ABC and `DiffusiveFlux(order)`
concrete class.  Key design decisions:

- `DiscreteOperator` gains abstract `order: int`; `NumericalFlux` inherits
  from it, narrowing the calling convention to cell-average → face-flux.
- `NumericalFlux.__call__(U: MeshFunction) → MeshFunction` (full-field):
  the returned MeshFunction is callable as `result((axis, idx_low))`.
  Full-field evaluation is JAX-friendly (one JIT-compiled array operation)
  and makes `NumericalFlux` a first-class `DiscreteOperator`.
- `DiffusiveFlux` derives stencil coefficients symbolically in `__init__`
  from the antisymmetric cell-average moment system — no hardcoded stencils.
  Validity declared as class attributes: `min_order=2`, `order_step=2`.
  The `__init__` guard is derived from these, making the constraint explicit.
- Convergence testing infrastructure: `tests/support/` provides a
  `CONVERGENCE_INSTANCES` registry and `CONVERGENT_ABCS` list.
  `conftest.py` enforces at collection time that every concrete
  `DiscreteOperator` subclass has instances registered.
  `test_convergence_order.py` auto-computes exact values via
  `CartesianRestrictionOperator(mesh, degree)(instance.continuous_operator(phi))`
  — no per-class oracle `error()` method is needed.

**C4 — `DiffusionOperator`, `continuous_operator`, `Discretization`. ✓**
Four additions delivered together:

1. `LazyMeshFunction[V](mesh, fn)` in `theory/discrete/` — callable-backed
   `MeshFunction`; generalizes the private `_FaceMeshFunction` that `DiffusiveFlux`
   previously returned.  All full-field `NumericalFlux.__call__` returns use it.
2. `DiffusionOperator` in `theory/continuous/` — concrete
   `DifferentialOperator[ZeroForm, OneForm]` representing `-d: Ω⁰ → Ω¹`.
   `_NegatedGradientField` moved here from `poisson_equation.py` and shared.
3. `DiscreteOperator` gains abstract `continuous_operator: DifferentialOperator`
   — the second falsifiable claim every discrete operator makes.
   `DiffusiveFlux.__init__` takes `continuous_operator` as a required
   constructor argument; the `__init__` guard enforces `isinstance(_, DiffusionOperator)`.
4. `FVMDiscretization(mesh, numerical_flux, boundary_condition)` —
   `Discretization` subclass in `geometry/`; `__call__(L)` produces
   `_AssembledFVMOperator` carrying `continuous_operator = L` from birth.
   `_AssembledFVMOperator.__call__(U)` computes
   `(1/|Ωᵢ|) · Σ_a [F(U)((a, i)) − F(U)((a, i−eₐ))]` lazily via `LazyMeshFunction`;
   mesh is read from `U.mesh` at call time (symbolic or concrete).
5. `CONVERGENT_ABCS = [DiscreteOperator]` replaces `[NumericalFlux]` as the
   single convergence root; `_AssembledFVMOperator` gets a `FVMDiscretizationOracle`
   that verifies `‖Lₕ Rₕ f − Rₕ L f‖_{∞,h} = O(hᵖ)` symbolically.

Lane C verified: commutation diagram at order p for `DiffusiveFlux(2)` and
`DiffusiveFlux(4)` via manufactured-solution polynomial on a 1D symbolic mesh,
registered in `tests/support/oracles/fvm_discretization.py`.  SPD derivation
deferred to C6.

**C5 — Automated convergence framework via `RestrictionOperator.degree`. ✓**
Complete the oracle-free convergence testing infrastructure:

1. `RestrictionOperator` gains abstract `degree: int` — the dimension of
   mesh elements being restricted to (n = cells, n−1 = faces, ...).
   This is the DEC cochain map parameter; `CartesianRestrictionOperator`
   gains a `degree` constructor argument, with existing cell-average
   behavior at `degree=n`.
2. `CartesianRestrictionOperator(mesh, degree=n−1)` — integrates a
   `OneForm`'s normal component over faces, producing a face-valued
   `MeshFunction`.  Uses the same SymPy integration machinery as the
   cell-average restriction.
3. `OneForm.component(axis)` — extract the a-th component as a `Field`,
   needed by the face restriction to compute the normal flux.
4. Oracle files deleted.  The convergence framework calls
   `RestrictionOperator(mesh, degree=n−1)(instance.continuous_operator(phi))`
   directly for the exact face flux.  `DiffusiveFluxOracle` and
   `CONVERGENCE_ORACLES` disappear; `CONVERGENT_ABCS = [DiscreteOperator]`
   is the single discovery root.

The commutation diagram for `NumericalFlux` is then closed formally:

```
φ ─────────(continuous_operator)──────▶ L(φ)
│                                           │
(CartesianRestrictionOperator, degree=n) (CartesianRestrictionOperator, degree=n−1)
│                                           │
▼                                           ▼
U_h ────────(NumericalFlux)─────────▶ F_h
```

**C6 — SPD analysis of the discrete Poisson operator.** For
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
   definiteness of `-∇²`. The explicit eigenvalues quoted in C7 are a
   consequence of SPD + translation invariance on `CartesianMesh`, not
   additional hypotheses.

The row ordering for matrix assembly is lexicographic
(idx → Σ_a idx[a]·N^a); unit-basis assembly `A eⱼ = Lₕ eⱼ` fills one
column per cell. Lane C verifies SPD symbolically at N = 4 in 1-D and
2-D for both `DiffusiveFlux(2)` and `DiffusiveFlux(4)`, so the assertion
does not depend on a numerical eigenvalue computation.

**C7 — `LinearSolver` hierarchy with `DenseJacobiSolver`. ✓** Introduced the
abstract `LinearSolver` interface, scoped explicitly to *linear* operators
(nonlinear problems need a separate `NonlinearSolver`). This PR develops
the interface, the `DenseJacobiSolver` implementation with matrix assembly
via unit basis, and the Jacobi spectral-radius derivation for `DiffusiveFlux(2)`.
The convergence-count Lane B check for order=4 is deferred to C8. The
derivation works simultaneously in two directions. Both directions are stated
for `FVMDiscretization(PoissonEquation, DiffusiveFlux(2), DirichletBC)` on
`CartesianMesh`; the same construction applies to `DiffusiveFlux(4)` but
the explicit spectral rate is different — see the "Order ≥ 4" remark below.

*Forward from the formal ingredients already in the code.* At the point C7
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
3. The SPD property of A, proved in C6's Lane C derivation (not asserted
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
symbols. SPD (from C6) still guarantees convergence qualitatively for
any α small enough, but the iteration-count bound must be re-derived
numerically by a one-off dense eigenvalue scan on a representative
grid. The empirical rate for `DiffusiveFlux(4)` is deferred to C8; the
closed-form spectral derivation is deferred and re-opened when multigrid
(Epoch 6) requires spectral bounds on wide-stencil operators.

All linear algebra is hand-rolled — no NumPy `linalg`, no LAPACK.

**C8 — DenseJacobiSolver convergence check (order=4). ✓** Lane C.
`DiffusiveFlux(4)` violates diagonal dominance: the interior stencil
`[1/12, −4/3, 5/2, −4/3, 1/12]/h²` has row-absolute-sum `17/6/h²` exceeding the
diagonal `5/2/h²`, so the Gershgorin bound on `λ_max(D⁻¹A)` is `32/15 > 2` and
standard Jacobi (`ω = 1`) diverges (spectral radius `17/15 > 1` at the
Nyquist mode). The fix: `DenseJacobiSolver` now derives `ω` automatically from
the Gershgorin bound `G = max_i Σ_j |A_{ij}/A_{ii}|` and sets `ω = min(2/G, 1)`.
For `DiffusiveFlux(2)` the interior bound is `G = 2`, giving `ω = 1` (standard
Jacobi, unchanged). For `DiffusiveFlux(4)`, `G = 32/15`, giving `ω = 15/16`.
Lane C verifies on an N = 8 1-D system that the solver reaches the prescribed
tolerance with monotonically decreasing residuals and an iteration count ≤ the
upper bound implied by the asymptotic convergence rate (tail geometric mean of
residual ratios), confirmed in `tests/test_convergence_order.py`.

**C9 — End-to-end convergence sweep.** Add `_ConvergenceRateClaim` to
`tests/test_convergence_order.py` to verify that the full solve pipeline
(FVMDiscretization + LinearSolver) recovers the correct convergence order
against a manufactured solution. The Lane C claim: the discrete solution
error `‖φ_h − Rₕ φ_exact‖_{L²_h}` converges at O(hᵖ) as h → 0, for every
`(solver, flux)` pair in the registries.

**Manufactured solution (1-D).** `φ(x) = sin(πx) + sin(3πx)` on [0, 1]
with homogeneous Dirichlet BC. Source: `ρ(x) = π² sin(πx) + 9π² sin(3πx)`.
Both modes have nonzero derivatives of all orders, so the leading
truncation-error term is excited for every `DiffusiveFlux(order)`. The
two-mode sum prevents the test field from being an eigenfunction of the
discrete operator. Note: 1-D suffices to verify convergence order at the
stencil level. Extension to a multi-dimensional sweep is deferred; see
open questions below.

**Registries.** Two new module-level registries alongside `_FLUXES`:

- `_SOLVERS`: all concrete `LinearSolver` instances in scope. Initially
  `[DenseJacobiSolver(tol=1e-8, max_iter=10_000)]`. Adding a new solver
  here automatically generates order-claims and solver-claims for every
  existing flux.
- `_CONVERGENCE_MESHES`: shared 1-D mesh sequence. N ∈ {8, 12, 16, 24, 32}
  — floor at N = 8 (below which the asymptotic regime is unreliable), five
  points for a robust log-log slope fit, tractable for both p = 2 and p = 4.

**`_ConvergenceRateClaim(solver, flux, meshes)`.** For each mesh in the
sequence, builds `FVMDiscretization(mesh, flux, DirichletBC(manifold))`,
sets `rhs = Rₕ ρ` via `CartesianRestrictionOperator`, solves with
`solver.solve(disc, rhs)`, and measures `‖φ_h − Rₕ φ_exact‖_{L²_h}`.
Fits a log-log slope over all five points and asserts slope ≥
`flux.order − 0.1`. Description:
`{SolverType}/{FluxType}(order={p})/convergence_rate`.

**`_CLAIMS` generation.** All four claim types are driven from the
registries; no individual claims are hand-listed:

```python
_CLAIMS = [
    *[_OrderClaim(f) for f in _FLUXES],
    *[_OrderClaim(FVMDiscretization(_dummy_mesh, f)()) for f in _FLUXES],
    *[_SolverClaim(s, f, _mesh_n8) for s in _SOLVERS for f in _FLUXES],
    *[_ConvergenceRateClaim(s, f, _CONVERGENCE_MESHES) for s in _SOLVERS for f in _FLUXES],
]
```

Adding a new `NumericalFlux` to `_FLUXES` or a new `LinearSolver` to
`_SOLVERS` automatically produces all four claim types for the new entry.

**Validation narrative (deferred).** A `validation/poisson/` narrative
application (manufactured-solution script, figures, Sphinx page) is deferred
pending a decision on the validation pattern for multi-dimensional problems.

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

3. **Multi-dimensional convergence sweep.** `_ConvergenceRateClaim` (C9)
   verifies convergence order in 1-D. A future item should extend the
   sweep to 2-D and 3-D `CartesianMesh` instances using the same registry
   pattern (`_FLUXES × _SOLVERS × _CONVERGENCE_MESHES`), once the compute
   budget and mesh-sequence floors for higher dimensions are established.
   The 2-D manufactured solution `φ(x, y) = sin(πx) + sin(3πx) +
   sin(πy) + sin(3πy)` (separable, homogeneous Dirichlet BC on the unit
   square) is the natural extension.

---

## Physics roadmap

### Foundation epochs

| Epoch | Layer | Capability |
|-------|-------|------------|
| 1 | Discrete | **Discrete operators and first Poisson solver.** `DivergenceFormEquation` hierarchy in `continuous/` (`PoissonEquation`). `Discretization` ABC + generic `FVMDiscretization(mesh, numerical_flux, boundary_condition)`. `NumericalFlux` family (`DiffusiveFlux` for Epoch 1; `HyperbolicFlux` for Epoch 4); order = min(reconstruction, face-quadrature, deconvolution) — all three independently verified. `LinearSolver` with `DenseJacobiSolver` (hand-rolled, dense, no LAPACK); scoped to linear operators only. Boundary conditions via discretization constructor. Truncation error proved symbolically; convergence verified against `sin(πx)sin(πy)`. FVM machinery reused in Epoch 4 by supplying a concrete Euler `DivergenceFormEquation` and swapping `NumericalFlux`; `LinearSolver` is *not* reused (Euler is nonlinear). |
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
