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

The codebase is organized into five packages with a strict dependency order:

```
theory/
  foundation/ ←── continuous/ ←── discrete/
                                        ↑         ↑
                                   geometry/   computation/
                                        ↑         ↑
                                        └─physics/─┘
```

`A ←── B` means B imports from A (B sits above A in the stack).
`computation/` has no imports from `theory/` or `geometry/`; the two
paths into `physics/` are independent.

`foundation/`, `continuous/`, and `discrete/` are nested under `theory/`,
making the symbolic-reasoning boundary a directory boundary.

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

**`physics/` is the application/concreteness layer.** It implements specific
physical models (PDE operators, discretization schemes). `physics/` may import
from all other packages.

**`computation/` is the numeric machinery layer.** It must not import from
`theory/` or `geometry/` or `physics/`; enforced by
`scripts/ci/check_computation_imports.py`. All numeric library imports
(`math`, `numpy`, `jax`, etc.) are confined here; enforced by
`scripts/ci/check_numeric_imports.py`.

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
        ├── ZeroForm                 — scalar field; degree = 0; codomain sympy.Expr
        ├── OneForm                  — covector field; degree = 1; codomain tuple[sympy.Expr, ...]
        ├── TwoForm                  — 2-form; degree = 2; codomain sympy.Matrix
        └── ThreeForm                — volume form; degree = 3; codomain sympy.Expr

DifferentialOperator(Function[Field, _C]) — L: Field → _C; interface: manifold, order
├── ExteriorDerivative                       — d: Ω^k → Ω^{k+1}; exact chain map on M.
│                                              degree=0: gradient  (ZeroForm  → OneForm)
│                                              degree=1: curl      (OneForm   → TwoForm,  3D only)
│                                              degree=2: divergence(TwoForm   → ThreeForm, n=3)
│                                              d∘d = 0 identically (exact sequence, no truncation error)
└── DivergenceFormEquation                   — ∇·F(U) = S in spatial-operator form;
                                               earned by: integral form ∮_∂Ωᵢ F·n dA = ∫_Ωᵢ S dV
                                               is fully determined by flux + divergence theorem,
                                               not derivable from bare DifferentialOperator.
                                               free: flux: Function[Field, TensorField], source: Field
                                               derived: order = 1
    └── PoissonEquation                      — -∇²φ = ρ; earned by: derived flux = -∇(·).
                                               The sign convention (flux = -∇φ, not +∇φ) ensures
                                               the discrete operator is positive definite.
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

### discrete/

**Horizontal mapping — every type in `continuous/` has an intended counterpart:**

| `continuous/` | `discrete/` | Notes |
|---|---|---|
| `TopologicalManifold` | `CellComplex` | topological space of cells |
| `Manifold` | `Mesh` | adds chart / coordinate geometry |
| *(none)* | `StructuredMesh` | regularity qualifier; no smooth analog |
| `Field[V]` | `DiscreteField[V]` | map from space to value |
| `ZeroForm` | `PointField[V]` | Ω⁰; point-valued field at mesh vertices (FD-style DOFs) |
| `OneForm` | `EdgeField[V]` | Ω¹; edge-integrated field (e.g. EMF in MHD constrained transport) |
| `TwoForm` | `FaceField[V]` | Ω²; face-integrated field; scalar flux F·n̂·|A| or matrix-valued |
| `ThreeForm` | `VolumeField[V]` | Ωⁿ (volume form); cell total-integral field (n-cochain) |
| `TensorField`, `SymmetricTensorField` | **missing** | rank-(p,q) annotated discrete fields; needed Epoch 7+ (rotating-frame metric, MHD) |
| `ExteriorDerivative` | `DiscreteExteriorDerivative` | exact chain map; d∘d=0; no truncation error |
| `DifferentialOperator` | `DiscreteOperator` | map between fields (approximation, O(hᵖ) error) |
| `DivergenceFormEquation` | — | bridge: `Discretization` maps a `DivergenceFormEquation` to a `DiscreteOperator` |
| `BoundaryCondition` | *(none)* | BC is a continuous concept; enters the discrete layer only through `Discretization` |
| *(none)* | `RestrictionOperator` | bridge concept: maps continuous `Field` → `DiscreteField`; no pure continuous analog |

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

DiscreteField(NumericFunction[Mesh, V])
                               — map from mesh elements to value type V;
                                  the discrete counterpart of Field.
                                  Earned by .mesh: Mesh typed accessor,
                                  parallel to Field.manifold.
                                  V is unconstrained: sympy.Expr for symbolic
                                  evaluation (order proofs), float for numeric
                                  paths, or any PythonBackend-compatible type.
├── PointField(DiscreteField[V])
│                              — abstract; Ω⁰ DOF location: values at mesh
│                                 vertices. Discrete counterpart of ZeroForm.
│                                 Indexed by vertex multi-index (i₀,…,iₙ₋₁);
│                                 vertex shape = cell shape + 1 per axis.
│                                 Natural DOF for finite-difference schemes.
│                                 Concrete subclass:
│                                   _CallablePointField — callable-backed (CartesianExteriorDerivative)
├── EdgeField(DiscreteField[V])
│                              — abstract; Ω¹ DOF location: values at mesh
│                                 edges. Discrete counterpart of OneForm.
│                                 Indexed by (tangent_axis, idx_low) mirroring
│                                 FaceField's (normal_axis, idx_low).
│                                 Natural DOF for the electric field E in MHD
│                                 constrained transport (Faraday: d: Ω¹ → Ω²).
│                                 Concrete subclass:
│                                   _CallableEdgeField — callable-backed (CartesianExteriorDerivative)
├── VolumeField(DiscreteField[V])
│                              — abstract; Ωⁿ DOF location: total integrals
│                                 ∫_Ωᵢ f dV over each cell (n-cochain).
│                                 Discrete counterpart of ThreeForm.
│                                 Concrete subclasses:
│                                   _CartesianVolumeIntegral — sympy totals (Rₕ)
│                                   _CallableVolumeField — callable-backed
└── FaceField(DiscreteField[V])
                               — abstract; Ω² DOF location: face-integrated
                                  values. Discrete counterpart of TwoForm.
                                  Indexed by (normal_axis, idx_low): axis ∈ [0, ndim)
                                  is the face normal; idx_low ∈ ℤⁿ is the
                                  low-side cell index.
                                    FaceField[scalar]        ↔ scalar flux F·n̂·|A|
                                    FaceField[sympy.Matrix]  ↔ matrix-valued flux
                                  The canonical return type of NumericalFlux.__call__
                                  and CartesianFaceRestriction.
                                  Concrete subclass:
                                    _CallableFaceField — callable-backed (NumericalFlux,
                                                         CartesianFaceRestriction)

RestrictionOperator(NumericFunction[F, DiscreteField[V]])
                               — free: mesh: Mesh;
                                  formal bridge from continuous/ to discrete/:
                                  a Function plus a Mesh yields a DiscreteField.
                                  F is a generic input type so that concrete
                                  subclasses can narrow it (e.g. ZeroForm, OneForm)
                                  without an LSP violation.  The output cochain
                                  level is fixed by the concrete subclass — the
                                  return type of __call__ encodes the DEC degree k,
                                  making a separate degree property redundant.

DiscreteBoundaryCondition(ABC)
                            — discrete counterpart of BoundaryCondition.
                              While BoundaryCondition describes the mathematical
                              constraint (φ|_∂Ω = g), DiscreteBoundaryCondition
                              describes how to extend a field beyond the mesh
                              boundary via ghost cells so that NumericalFlux
                              stencils can be evaluated at boundary-adjacent cells.
                              Abstract: extend(field, mesh) → DiscreteField
                              Concrete subclasses:
                                DirichletGhostCells — odd reflection (φ = 0 at face)
                                PeriodicGhostCells  — wrap-around (φ(x+L) = φ(x))

Discretization(ABC)           — free: mesh: Mesh, boundary_condition: DiscreteBoundaryCondition
                              Encapsulates the scheme choice (reconstruction,
                              numerical flux, quadrature, boundary condition).
                              __call__(self) → DiscreteOperator produces the
                              assembled Lₕ that makes the commutation diagram
                                Lₕ ∘ Rₕ ≈ Rₕ ∘ L   (up to O(hᵖ))
                              hold, interpreted on test fields f ∈ C^{p+2}(Ω);
                              "≈" means ‖Lₕ Rₕ f − Rₕ L f‖_{∞,h} = O(hᵖ)
                              as h → 0, measured in the local ℓ∞ norm over
                              interior cells.  The approximation order p is a
                              property of the concrete scheme, proved by its
                              convergence test — not a parameter of the
                              abstract interface.
                              The commutation check verified algebraically via
                              SymPy is the machine-checkable derivation required
                              by Lanes B and C.
                              Formally separate from Rₕ: Rₕ projects field values
                              (Function → DiscreteField); Discretization projects
                              operators (DivergenceFormEquation → DiscreteOperator).

DiscreteOperator(NumericFunction[_In, _Out])
                            — discrete operator parameterized by input and
                              output types.  Subclasses fix the cochain shape:
                                Discretization: DiscreteField → DiscreteField
                                NumericalFlux:  DiscreteField → FaceField
                              Earns its class via two falsifiable claims:
                                order: int — composite convergence order
                                continuous_operator: DifferentialOperator —
                                  the continuous operator this approximates
                                  (threaded automatically by Discretization
                                  from its input L)
                              Not independently constructed from stencil
                              coefficients; produced by a Discretization.

NumericalFlux(DiscreteOperator[DiscreteField, FaceField])
                            — cell-average → face-flux operator:
                                __call__(U: DiscreteField) → FaceField
                              where U holds cell-average values.  The
                              returned FaceField is indexed as
                              result((axis, idx_low)) and returns the flux
                              F·n̂·|face_area| at that face.  Inherits order
                              and continuous_operator from DiscreteOperator.

DiscreteExteriorDerivative(ABC)
                            — NOT a DiscreteOperator; exact chain map, no truncation
                              error. Interface: mesh: Mesh, degree: int,
                              __call__(field: DiscreteField) → DiscreteField.
                              d∘d = 0 exactly (algebraic identity).
                              Does not carry order or continuous_operator because
                              it is not an approximation — it is exact by construction.
```

### geometry/

Pure geometric objects and geometric operations on them.
Symbolic-reasoning layer: no numeric library imports.

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

CartesianRestrictionOperator(RestrictionOperator[F, sympy.Expr])
                                       — abstract base for all Rₕᵏ on CartesianMesh.
                                         Encodes the two Cartesian invariants: mesh is
                                         CartesianMesh; output value type is sympy.Expr.
                                         A future non-Cartesian geometry provides a
                                         parallel abstract base (same structure, different
                                         mesh type and value type).
├── CartesianVolumeRestriction(CartesianRestrictionOperator[ZeroForm])
│                                      — Rₕⁿ: ZeroForm → VolumeField (∫_Ωᵢ f dV, total)
│                                        In Cartesian coords dV=1, so ZeroForm integrates
│                                        directly as scalar density; no n-form wrapping.
│                                        FV restriction: cell-average DOF choice.
├── CartesianFaceRestriction(CartesianRestrictionOperator[DifferentialForm])
│                                      — Rₕⁿ⁻¹: DifferentialForm → FaceField
│                                        Abstract input is the (n-1)-form; the Cartesian
│                                        representation uses OneForm as proxy (Hodge
│                                        isomorphism in flat space): F.component(a)
│                                        gives the face-normal flux density at all dims.
│                                        ∫_{transverse} F.component(a)|_{x_a=face} dx_⊥
├── CartesianEdgeRestriction(CartesianRestrictionOperator[OneForm])
│                                      — Rₕ¹: OneForm → EdgeField (edge line integral)
│                                        OneForm is dimension-independent here: Rₕ¹
│                                        always integrates a 1-form along 1-D edges.
└── CartesianPointRestriction(CartesianRestrictionOperator[ZeroForm])
                                       — Rₕ⁰: ZeroForm → PointField (cell-center eval)
                                         ZeroForm is dimension-independent: Rₕ⁰ always
                                         evaluates a scalar at points.
                                         FD restriction: point-value DOF choice.
                                         Commutation: Dₖ ∘ Rₕᵏ = Rₕᵏ⁺¹ ∘ dₖ holds exactly
                                         for all k (FTC for k=0; Stokes for k=1)

CartesianExteriorDerivative(DiscreteExteriorDerivative)
                                       — exact discrete exterior derivative on CartesianMesh.
                                         degree=0: (d₀φ)(a,v) = φ(v+eₐ) − φ(v)   (gradient)
                                         degree=1: Yee-grid curl (3D only)
                                           (d₁A)(a,c): boundary circulation of A
                                           around the face with normal axis a
                                         degree=2: (d₂F)(c) = Σₐ[F(a,c)−F(a,c−eₐ)] (divergence)
                                         d_{k+1}∘d_k = 0 exactly for all k.
```

### physics/

Concrete PDE model implementations and simulation state.
Application/concreteness layer: may import from all other packages.

```
NumericalFlux implementations:
├── DiffusiveFlux(order)       — F(U) = −∇U; stencil coefficients derived
│                                 symbolically in __init__ from the antisymmetric
│                                 cell-average moment system.
│                                 Validity: min_order=2, order_step=2 (even orders
│                                 only; antisymmetric design kills odd error terms).
│                                 One class, not one per order: DiffusiveFlux(2)
│                                 and DiffusiveFlux(4) are instances, not subclasses.
├── AdvectiveFlux(order)       — F(U) = v·U; symmetric centered reconstruction.
└── AdvectionDiffusionFlux(order)
                               — F(U) = U − κ∇U; combines advective and diffusive
                                 parts at unit Péclet number.

DivergenceFormDiscretization(Discretization)
                               — free: numerical_flux, boundary_condition
                                 Discretization of a linear operator L = ∇·f via
                                 the divergence-form factorization.  Given a
                                 NumericalFlux discretizing f: state → face values,
                                 builds Lₕ = (1/vol) · d_{n−1} ∘ F̂ ∘ bc.extend.
                                 The "flux" is a formal intermediate at faces;
                                 the equations we currently solve (Poisson, steady
                                 advection, steady advection-diffusion) are elliptic
                                 algebraic constraints, not time evolutions.
                                 Specializations belong in the NumericalFlux —
                                 not in a new Discretization subclass per equation.
```

### computation/

The only layer that may import numeric libraries (`math`, `numpy`, `jax`,
etc.); all other layers are restricted to the Python standard library and
approved symbolic packages. Enforced by `scripts/ci/check_numeric_imports.py`.
Must not import from `theory/`, `geometry/`, or `physics/`; enforced by
`scripts/ci/check_computation_imports.py`.

```
Real(Protocol)      — scalar numeric protocol; satisfied by float, int,
                      numpy.float16/32/64, JAX scalars. Covers exactly the
                      arithmetic operations Tensor applies to its elements.

Tensor              — arbitrary-rank numeric array backed by a pluggable
                      Backend. Single public API over multiple storage
                      strategies. Supports construction, indexing,
                      arithmetic (+, −, *, /), einsum, matmul, norm, diag,
                      SVD, copy, to_list, and to(backend). Rank-0 through
                      rank-n; all shapes uniform (no jagged arrays).

Backend(Protocol)   — per-instance dispatch strategy. Mixed-backend
                      arithmetic raises ValueError. Backends:

    PythonBackend   — nested Python lists; reference implementation;
                      no external dependencies. Leaf values are unconstrained
                      Python objects, so sympy.Expr leaves work transparently
                      (used by the symbolic order-proof path in physics/).
    NumpyBackend(dtype=None)
                    — NumPy ndarray; dtype inferred from input by default
                      or fixed to an explicit numpy dtype; vectorized via
                      BLAS/LAPACK.
    JaxBackend      — JAX array; immutable functional updates routed through
                      `Tensor.__setitem__` via `slice_set`. Caller is responsible
                      for `@jax.jit` placement at solver / time-step granularity.

LinearSolver        — mesh-agnostic interface: solve(a: Tensor, b: Tensor) → Tensor.
                      Accepts an assembled N×N stiffness matrix and an N-vector
                      RHS; returns the solution vector. Assembly and index mapping
                      are the caller's responsibility, keeping computation/ free
                      of theory/discrete/ and physics/ dependencies.
                      SCOPE: linear operators only. Epoch 6 hydro (nonlinear
                      flux) requires a separate NonlinearSolver / Newton
                      iteration. LinearSolver is not the shared machinery
                      for Epoch 6; only DivergenceFormDiscretization and NumericalFlux
                      are reused across epochs.
                      Ships DenseJacobiSolver (weighted Jacobi, ω derived
                      from Gershgorin bound; works for both order=2 and
                      order=4 stencils) and DenseLUSolver (direct, in-place
                      LU with partial pivoting). Both operate on Tensor;
                      linear algebra hand-rolled, no LAPACK. Convergence
                      tests cap at N ≤ 32 in 2-D (≤ 1024 unknowns).
```

**Planned additions (Epoch 4):** time-integration layer — `TimeIntegrator`
ABC, `RungeKuttaIntegrator`, `TimeStepper`, `Autotuner` extension; structured
RHS protocols (`HamiltonianSplit`, `WithJacobian`, `Additive`,
`LinearPlusNonlinear`, `OperatorSplit`) and additional integrator families
introduced phase-by-phase per the [Epoch 4
sprint](#current-work-epoch-4--time-integration-layer).

### Cross-cutting

**Kernel composition model.**
Realized as the `Backend` protocol in `computation/backends/`. `Tensor`
is the single public interface; `Backend` is the strategy that governs
storage and execution. Backends are per-instance; dispatch is
identity-checked (`is`), not type-checked, so user-defined backends
satisfying the protocol are first-class. Open questions: `@jax.jit`
tracing policy for `JaxBackend` (per-call JIT vs. solver-level JIT);
whether `set_default_backend` is sufficient for solver-level selection
or a solver-level override is needed.

---

## Current work: Epoch 4 — Time integration layer

**Target.** A typed, modular time-integration layer that scales from explicit
Runge-Kutta to implicit, exponential, multistep, variable-order, splitting,
and symplectic methods without interface redesign.  The interface is shaped
by a six-axis DSL — RHS protocol, state representation, step program,
coefficient algebra, controller, verification primitives — and the roadmap
relaxes each axis from its degenerate value through nine phases tied to
physics-epoch needs.

The DSL framing rests on the observation that "basic arithmetic + black-box
RHS" is provably the Butcher group (B-series): every richer family —
exponential, IMEX, Rosenbrock, multistep, variable-order, symplectic —
relaxes a specific structural constraint.  Each phase below relaxes exactly
one or two of those constraints with concrete physics motivation, so the
DSL is earned axis-by-axis rather than designed up-front.  The typed slot
shape (`RHSProtocol`, integrator-specific `State`, `Controller`) is
established in `computation/time_integrators/`; subsequent phases extend
without breaking interfaces.

**Phase 6 — Explicit Adams-Bashforth.** `ABState(t, u, history)` carries the
current solution and the function-value history as an immutable tuple, most
recent first; `AdamsBashforthIntegrator(beta, order)` applies the AB weights
and bootstraps the first k−1 steps with RK4.  Named instances: `ab2` (order 2),
`ab3` (order 3), `ab4` (order 4).  Verification: convergence slope on dy/dt = λy;
abundance conservation on the A→B→C decay chain — `JacobianRHS` satisfies
`RHSProtocol`, so the same problem object is shared with the DIRK claims without
exposing the Jacobian to the AB integrator; hard conservation check
`|Σ Xᵢ − 1| < 1e-12` passes because zero column-sum rate matrices are conserved
by any linear combination of past function values.  First phase with non-trivial
typed state.  Mostly groundwork for Phases 7a–12.

**Phase 7a — Nordsieck state + BDF fixed-order integrators.**
`NordsieckState(t, h, z)` carrying the Nordsieck vector
`z = [y, hy', h²y''/2!, …, hᵍy⁽ᵍ⁾/q!]` with step size `h` baked in so
prediction is a pure Pascal-matrix multiply; `BDFFamily(q_max=6)` as a
parametric coefficient provider producing the BDF `α`/`l` vectors for any
order `q`; `NordsieckIntegrator(family, q)` with a Newton corrector
(BDF is implicit, requires `WithJacobianRHSProtocol`); bootstrap via RK4
for the first `q` steps followed by Nordsieck vector initialization; named
instances `bdf1`–`bdf4`.  Verification: convergence claims for BDF orders
1–4 on the A→B→C abundance network; hard conservation check
`|ΣXᵢ − 1| < 1e-12`.

**Phase 8 — Adams family + fixed-order Adams integrators.**
`AdamsFamily(q_max=12)` as a parametric coefficient provider for
Adams-Moulton orders 1–12; Adams correction mode added to
`NordsieckIntegrator` using fixed-point iteration (no Jacobian required,
satisfies plain `RHSProtocol`); named instances `adams_moulton1`–
`adams_moulton4`.  Verification: convergence claims for Adams orders 1–4
on the same abundance network; same conservation check.

**Phase 9 — Nordsieck order-change rescaling.** The pure mathematical
primitive for changing order within a fixed family: a function (or method
on `NordsieckState`) that remaps the history vector when `q` shifts —
padding with zero when raising, truncating when lowering — together with
the accompanying step-size rescaling formula derived from the Nordsieck
representation.  Verification: round-trip property test (raise order then
immediately lower it and confirm `z[0]` is unchanged to round-trip
accuracy); confirm that a rescaled state integrates with the same accuracy
as a fresh `init_state` at the new `q`.

**Phase 10 — Variable-order within a family.** `OrderSelector` estimating
truncation error at orders `q−1`, `q`, `q+1` from the corrector residual
and Nordsieck history differences, selecting `q_next` to minimize work per
unit accuracy; `VariableOrderNordsieckIntegrator` wrapping a fixed family
and applying the Phase 9 order-change rescaling when `q` shifts;
step-size rescaling on rejection using the LTE estimate at current `q`.
Verification: order-selection claim on a non-stiff 3-species network
(λ_stiff/λ_slow ~ 10); confirm `q` climbs to the expected ceiling on
smooth data and drops when the solution sharpens.

**Phase 11 — Stiffness detection + family-switch transformation.**
`StiffnessSwitcher` carrying a streaming spectral-radius estimate from the
Newton Jacobian (Gershgorin bound on the last accepted `hJ`), triggering
Adams→BDF when the dominant eigenvalue magnitude exceeds a threshold and
BDF→Adams when it falls back; family-switch Nordsieck transformation
remapping the history vector between BDF and Adams `α`/`l` bases.
No full controller yet.  Verification: invertibility test (`Adams → BDF →
Adams` restores `z[0]` to round-trip accuracy); stiffness detector fires
at the correct eigenvalue magnitude threshold.

**Phase 12 — VODEController.** `VODEController` composing `OrderSelector`
(Phase 10) and `StiffnessSwitcher` (Phase 11) into a full VODE-style
adaptive controller.  Verification: claim on a network with a deliberate
fast/slow species pair (λ_stiff/λ_slow ~ 10³); Adams must be active
during the slow early transient and BDF must engage once the fast
timescale dominates.  Activates the "method family + policy" axis — the
integrator's identity becomes a function of run-time diagnostics, not a
fixed tableau.

**Phase 13 — Exponential integrators.** `LinearPlusNonlinearRHS(L, N)`
protocol; φ-function evaluation (scaling-and-squaring on dense `hL` for
small problems, Krylov / Arnoldi projection for large problems);
`PhiFunction(k)` coefficient algebra (operator-valued tableau entries);
named instances: ETD-Euler, ETDRK2, ETDRK4 (Cox-Matthews), Krogstad's
method.  Verification: stiff-order conditions (Hochbruck-Ostermann) on
`a_ij(z), b_j(z)` as functions of `z`; convergence on Allen-Cahn /
Gray-Scott / 2D Burgers.  Activates the coefficient-algebra axis.
Epoch 11 radiation enabler.

**Phase 14 — Operator splitting (Strang / Lie).** `OperatorSplitRHS([f_1, …,
f_k])` protocol; `StrangSplittingIntegrator(sub_integrators, sequence)` as
a meta-integrator delegating each substep to a peer `TimeIntegrator`.
Verification: combined order from commutator analysis (Lie 1st order;
Strang 2nd order; Yoshida-style triple-jump for higher even orders).
Activates compositionality of the integrator stack itself.  Epoch 10 MHD /
multi-physics enabler.

**Cross-phase deliverable: integrator benchmark library.** A growing
problem catalog and harness against which integrators are graded on
correctness, work-per-accuracy, scaling with state dimension, and
conservation-law fidelity.  Structured into three tiers that activate
as the corresponding integrator families come online; the harness
skeleton and Tier A ship alongside Phase 0 so that subsequent phases
each land with a benchmarkable deliverable.  Everything in the library
is Lane C: every problem RHS is coded from first-principles ODE
definitions, every rate coefficient is generated procedurally, and
every reference solution is computed inline by our own integrators
at tightened tolerance.  No external code is ported, no
rate-coefficient or trajectory tables are loaded from disk, and no
external integrator is invoked at any point.

*Tier A — Non-stiff (with Phase 0–1).* Scalar linear `dy/dt = λy`
(promoted from the temporal-convergence claim); logistic;
two-species predator-prey (conserved quantity, no closed form);
three-compartment SIR; two-body gravitational (analytical orbit with
energy and angular-momentum invariants); three-equation chaotic
system (integrator-sensitivity probe); two-species reaction-diffusion
oscillator.

*Tier B — Stiff baselines (with Phase 4).* Parametric synthetic
problems exercising the stiffness regimes that real applications
present:

- Stiff scalar with forcing, `dy/dt = -k · y + s(t)`, with
  `k ∈ {10, 10², 10⁴, 10⁶}` — stability-detection probe.
- Three-species mass-conserving stiff chemistry archetype, with
  parameterized rate disparity (up to ~10⁹ between fast and slow
  timescales).
- Synthetic small-and-medium stiff networks of `n ∈ {8, 20}` species,
  produced by the Tier C generator at small sizes and tuned for
  stiffness contrast rather than nuclear-network resemblance.
- Stiff nonlinear oscillator `du/dt = v, dv/dt = -ω² u − α(u² − 1) v`
  with `α ∈ {1, 10², 10³, 10⁵}` as a fixed-problem stiffness sweep.

*Tier C — Network-scale, nuclear-resembling (with Phases 7a–12).* The
user-facing payload.  Synthetic networks designed to numerically
resemble thermonuclear reaction networks: dozens to hundreds of
species, rate coefficients spanning ~10 orders of magnitude, sparse
Jacobian, exact mass conservation, positivity of every species.
Concrete problems:

- **Synthetic alpha-chain networks**, parameterized by chain length
  `n ∈ {13, 19, 21, 51, 101, 201, 501}`.  Linear capture-chain
  topology emulating alpha-rich nuclear burning; rate coefficients
  drawn from a log-normal distribution to span ~10 decades.  Exact
  invariants by construction: `Σ X_i = 1` (mass conservation) and
  `X_i ≥ 0` (positivity).
- **Synthetic CNO-cycle networks**, `n ∈ {6, 12, 24}`.  Cyclic
  topology with parameterized breakout-branch ratios; tunable
  stiffness contrast.
- **Synthetic rp-process networks**, `n ∈ {30, 60, 120}`.  Branched
  topology emulating proton-rich freezeout structure with multiple
  competing timescales.

*Synthetic network generator.* Small DSL for declaring
reaction-network topologies (alpha-chain, cycle, branched, random
sparse), parameterized by species count and rate-coefficient
distribution.  Produces a `WithJacobianRHS` instance with sparse
(CSR) Jacobian and a linear conservation matrix `C` such that
`C · Y = 0` is an exact algebraic identity.  Random-rate draws are
seeded so problem definitions are reproducible across runs.

*Metrics.* Per `(integrator, problem)` cell: final-time L² and L∞
error vs reference; RHS and Jacobian evaluation counts; wall time;
accepted and rejected step counts; conservation drift
`‖C · Y(t_end)‖`; minimum `X_i` over the trajectory (positivity);
for Tier C, runtime scaling exponent in `n`.

*Harness.* Lives in `tests/benchmarks/`, separate from the
correctness claims in `tests/test_time_integrators.py`.  Auto-discovers
`BenchmarkProblem` instances and runs each `TimeIntegrator` from the
`_INSTANCES` registry in `tests/test_time_integrators.py` against every
compatible problem (compatibility = problem's `RHSProtocol` satisfies
the integrator's `requires_rhs`).  Emits a JSON report.  Run under an
opt-in `@pytest.mark.benchmark` so day-to-day CI is unaffected; a
dedicated GitHub Actions workflow runs the suite on a schedule and
surfaces regressions by comparing each run against the prior workflow
run's report.

*Reference solutions are computed inline.*  Where an analytical
solution exists (linear, two-body, etc.), it is the reference.  Where
no analytical solution exists, the reference is computed at test time
by running an integrator from `_INSTANCES` at very tight tolerance.
A session-scoped pytest fixture caches the result so the cost is paid
once per session, not once per parameterized run.  To mitigate the
bootstrap risk of using the integrator family under test to verify
itself, the reference integrator is chosen from a *different family
or order* than the integrator under test wherever the integrator
stack permits: a higher-order RK references a lower-order RK; once
Phase 4 lands, an implicit integrator references an explicit one on
stiff problems where both apply; once multiple families exist they
cross-validate.  For phases where only one family is available,
bootstrap verification is supplemented by the convergence-order
claims from Phase 2 — a method that satisfies its symbolic order
conditions and converges at the predicted rate is unlikely to harbor
a family-wide bug invisible to self-reference.

**Open questions — Epoch 4 design points:**

1. **Coefficient-algebra typing for Phase 13.** Whether to parameterize
   `RungeKuttaIntegrator` over a `Coefficient` type variable (uniform
   surface, every method has the same class) or to introduce
   `ExponentialRKIntegrator` as a sibling class (more readable for the
   common case where coefficients are rationals).  Decision deferred until
   Phase 4 / 5 stabilize the implicit/explicit type story.

2. **Autotuner generalization.** Does the static autotuner from Phase 0
   (`Constant` controller + descriptor-driven `recommended_dt`) survive
   once VODE-style controllers exist in Phase 12, or does Phase 12 subsume
   the static autotuner as the trivial constant-policy case?  Resolves in
   Phase 12.

3. **Stiffness detector reuse.** Phase 11 needs an online ρ(J) estimate;
   Phase 13 may benefit from the same machinery (deciding when `‖hL‖` is
   large enough to justify exponential treatment over fully-explicit).
   Factor into a shared `StiffnessDiagnostic` from Phase 11 onward.

4. **`set_default_backend` vs. solver-level override.** Carried forward
   from the prior epoch.  Time-integrator code must inherit whichever
   resolution lands; a per-`TimeStepper` backend override is the natural
   API extension if process-wide defaults turn out to be insufficient.

**Open questions carried forward into Epoch 4:**

- AMR (Epoch 12): time-stepper must accept hierarchical state once meshes
  refine; integrator state types may need a coarse-fine variant.
- GR (Epoch 13): integrator state and step program become coupled to a
  dynamical 3-metric; the structured-RHS protocol may need a
  `WithMetricRHS` variant.
- Temporal convergence verification across spatial dimensions: spatial
  framework now covers 1D / 2D / 3D; temporal claims should likewise
  verify across problem dimensions where applicable.

---

## Cross-cutting open questions

Items that are not scoped to any specific epoch.  They surface here so they
are not lost; the decision of when and how to schedule them is made when
the implementation lane becomes clear.

**HDF5 checkpoint/restart.** Persistent serialization of simulation state
(time-stepper state, mesh fields, parameters) with provenance sidecars
(git hash, timestamp, parameter record).  GPU-written checkpoints must be
readable on CPU-only machines.  Not assigned to an epoch; the right time
to schedule depends on which integrator state types end up carrying
non-serializable data and on whether checkpointing arrives as part of a
broader persistence layer or as a standalone capability.

---

## Physics roadmap

### Foundation epochs

| Epoch | Layer | Capability |
|-------|-------|------------|
| 0 | Theory / Geometry | **Mathematical foundations. ✓** Layer architecture and symbolic-reasoning import boundary; `foundation/`, `continuous/`, `discrete/`, `geometry/` type hierarchies; `CellComplex`, `Mesh`, `StructuredMesh`, `DiscreteField`, `VolumeField`, `RestrictionOperator`; process discipline M0–M2. |
| 1 | Geometry / Validation | **Observational grounding. ✓** `EuclideanManifold`, `CartesianChart`, `CartesianMesh`; first `validation/` notebook (Schwarzschild spacetime, GPS time dilation); settles `SymbolicFunction` interface and `Point` type (M3). |
| 2 | Discrete | **FVM Poisson solver. ✓** `PoissonEquation`; `DiffusiveFlux(2,4)`; `DivergenceFormDiscretization` + `NumericalFlux` family; oracle-free convergence framework; SPD analysis; `LinearSolver` ABC with `DenseJacobiSolver` and `DenseLUSolver`; end-to-end O(hᵖ) convergence sweep. FVM machinery reused from Epoch 6 onward. |
| 3 | Computation | **Backend-agnostic computation layer. ✓** `Tensor` (arbitrary rank, `Real` protocol); `Backend` protocol with `PythonBackend`, `NumpyBackend`, `JaxBackend`; mixed-backend arithmetic guards; AST-based numeric-import boundary; self-calibrating roofline performance gate; `LazyDiscreteField` collapsed into `FaceField` and `_BasisField`. |
| 4 | Computation | **Time integration layer.** Six-axis DSL (RHS protocol, state, step program, coefficient algebra, controller, verification primitives) with explicit RK as the first instantiation; phases extend to adaptive control, B-series verification, symplectic, implicit, IMEX, multistep, variable-order, exponential, and splitting families. In progress. |

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

### Per-epoch verification standard

Every epoch must satisfy this checklist before it is considered verified:

- Derivation document with SymPy checks for any new numerical scheme (Lanes B and C)
- At least one externally-grounded convergence test against an analytical solution
  or observational data (not an engine-generated golden file); where an analytical
  solution exists, the relevant `NumericFunction.symbolic` is declared so the
  check runs automatically
- Lane A/B/C classification stated in the PR description
