# Cosmic Foundry — Architecture

## Startup context

Read this section and [## Current work](#current-work) at session start.
Do not read this file end-to-end unless the task requires a broad architecture
review; use the anchors below to load only the sections that govern the files
being changed.

- For foundational project claims, read [## Commitments](#commitments).
- For package dependency rules and type hierarchy details, read
  [## Layer architecture](#layer-architecture).
- For deferred cross-cutting decisions, read
  [## Cross-cutting open questions](#cross-cutting-open-questions).
- For time-integrator design, read
  [## Epoch 4 — Time integration verification (complete)](#epoch-4--time-integration-verification-complete).
- For long-horizon sequencing, read [## Physics roadmap](#physics-roadmap).

This section is a navigation index only.  Architectural facts live in their
own sections so that each decision has one canonical home.

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

**Time-integration layer** (`computation/time_integrators/`).  A typed,
modular layer supporting explicit RK, implicit DIRK, IMEX, exponential,
multistep (Adams / BDF), variable-order, symplectic, and operator-splitting
families through a common six-axis DSL (RHS protocol, state, step program,
coefficient algebra, controller, verification primitives).

```
RHS protocols — each narrows RHSProtocol to expose structure the integrator exploits:

RHSProtocol                      — base: __call__(t, u) → Tensor
├── BlackBoxRHS                  — wraps any callable
├── JacobianRHS                  — adds .jac(t, u) for Newton-based methods
├── FiniteDiffJacobianRHS        — finite-difference Jacobian approximation
├── SplitRHS                     — (explicit, implicit) split for ARK
├── HamiltonianRHS               — (dH_dq, dH_dp) for symplectic methods
├── SemilinearRHS                — (L, N) split for exponential integrators
└── CompositeRHS                 — [f_1, …, f_k] for operator splitting
                                   (SplittingStep sequence drives substep weights)

State types:

ODEState(NamedTuple)             — (t, u, dt, err, history); unified state type
                                   used by all integrators; history is None for
                                   single-step methods, tuple[Tensor, ...] for
                                   explicit multistep (Adams-Bashforth), and
                                   NordsieckHistory for Nordsieck-form methods
NordsieckHistory                 — Nordsieck vector (z, h) with rescale_step()
                                   and change_order(); stored in ODEState.history

Integrators:

RungeKuttaIntegrator             — Butcher-tableau explicit RK (orders 1–6)
                                   instances: forward_euler(1), midpoint(2), heun(2),
                                   ralston(2), rk4(4), bogacki_shampine(3,embedded),
                                   dormand_prince(5,embedded), butcher_6(6)
ImplicitRungeKuttaIntegrator     — implicit RK
                                   instances: backward_euler(1), implicit_midpoint(2),
                                   crouzeix_3(3), gauss_legendre_2_stage(4),
                                   radau_iia_3_stage(5), gauss_legendre_3_stage(6)
AdditiveRungeKuttaIntegrator     — additive RK (paired explicit + implicit tableaux)
                                   instances: imex_euler(1), ars222(2),
                                   imex_ssp3_433(3), ark436_l2sa(4)
ExplicitMultistepIntegrator      — explicit linear multistep (Adams-Bashforth)
                                   instances: ab1, ab2, ab3, ab4, ab5, ab6
MultistepIntegrator              — fixed-order Nordsieck-form BDF / Adams-Moulton
                                   factories: bdf_family → bdf1–bdf6
                                              adams_family → adams_moulton1–adams_moulton6
VODEController                   — adaptive Nordsieck controller combining
                                   OrderSelector and StiffnessSwitcher
LawsonRungeKuttaIntegrator       — integrating-factor RK for semilinear systems
                                   instances: lawson_rk1–lawson_rk6
SymplecticCompositionIntegrator  — position-Verlet family for separable Hamiltonian
                                   systems; inherits TimeIntegrator; accepts
                                   HamiltonianRHS with split_index
                                   instances: symplectic_euler(1), leapfrog(2),
                                   forest_ruth(4), yoshida_6(6)
CompositionIntegrator            — meta-integrator composing sub-integrators;
                                   factories: lie_steps()(1), strang_steps()(2),
                                   yoshida4_steps()(4, negative substep weights),
                                   yoshida6_steps()(6, negative substep weights)

Controllers:

ConstantStep                     — fixed step size
PIController                     — Gustafsson PI formula with accept/reject
OrderSelector                    — Nordsieck order and step-size policy
StiffnessSwitcher                — Adams/BDF family-switch policy

Infrastructure:

Integrator                      — drives integrator + controller loop;
                                   advance(rhs, u0, t0, t_end) → ODEState
PhiFunction(k)                   — φ_k operator action for exponential methods
StiffnessDiagnostic              — online spectral radius estimation
Tree / elementary_weight / trees_up_to_order
                                 — B-series order-condition verification
```

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

## Epoch 4 — Time integration verification (complete)

The nuclear astrophysics stress-test sprint (F1–F5) is complete.  The
infrastructure below is the foundation for Epoch 9 microphysics work.

#### Design principle

`AutoIntegrator` is the single correct entry point.  The caller expresses
the mathematical structure of the RHS via the appropriate protocol;
`AutoIntegrator` dispatches to the correct family from there.  No phase
ever calls an explicit integrator on a problem whose RHS structure calls
for an implicit one.

---

#### `ReactionNetworkRHS` protocol

A new RHS protocol sitting above `JacobianRHS` in the hierarchy, exposing
the factored structure of a system of paired forward/reverse reactions:

```
ReactionNetworkRHS(JacobianRHS)

    stoichiometry_matrix          # S: (n_species × n_reactions), integer
    forward_rate(t, X) → Tensor   # r⁺: n_reactions-vector, ≥ 0 for X ≥ 0
    reverse_rate(t, X) → Tensor   # r⁻: derived from r⁺ via detailed balance,
                                  #     not independently specified

    # Enforced at construction:
    # - reverse_rate is computed from forward_rate and thermodynamic data
    #   (partition functions / binding energies / free energies) via the
    #   detailed balance relation.  This guarantees the fully-equilibrated
    #   network recovers the correct thermodynamic fixed point.
    # - forward_rate(t, X) ≥ 0 for all t, X ≥ 0.

    # Derived at construction, not recomputed at runtime:
    conservation_basis            # left null space of S;
                                  # shape (n_conserved, n_species)
    conservation_targets          # w·X₀ for each conservation row w
    constraint_basis              # independent subset of the m pairwise
                                  # equilibrium conditions {r⁺ⱼ = r⁻ⱼ};
                                  # rank ≤ n_species − n_conserved

    # Implied interface:
    # __call__(t, X) = S @ (r⁺(t, X) − r⁻(t, X))
```

`AutoIntegrator` checks `ReactionNetworkRHS` before `JacobianRHS` (subtype
specificity), routing to the constraint-aware path rather than the plain
implicit-RK path.

The protocol is not nuclear-specific.  Any system of coupled forward/reverse
reactions — chemical kinetics, nuclear burning, radiative processes — satisfies
it.  The stoichiometry analysis is identical regardless of what the species
physically are.

---

#### New integrator infrastructure

Four additions to the time integration layer, each introduced in the
corresponding problem phase.  `TimeIntegrator.step(rhs, state, dt) → ODEState`
does not change signature; the new machinery lives in the state type,
the Newton kernel, and the controller.

**Conservation projection** (introduced in F2).  A free function
`project_conserved(X, basis, targets) → Tensor` returning the nearest point
in the conservation hyperplane {X : basis · X = targets}.  The projection is
orthogonal: X′ = X − basisᵀ (basis basisᵀ)⁻¹ (basis · X − targets).
Cost is O(n_conserved² · n_species); applied once per accepted step by
the controller.

**Constraint activation state in `ODEState`** (introduced in F4).  A new
optional field `active_constraints: frozenset[int] | None` on `ODEState`.
`None` (the default for all existing code) means no constraint tracking.
A frozenset of reaction-pair indices means those pairs are currently treated
as algebraic constraints.  The integrator passes this field through without
interpreting it; the controller and RHS read and write it.

**Projected Newton iteration** (introduced in F3).  `newton_solve` gains
an optional `constraint_gradients: Tensor | None` argument (shape
k × n_species, the gradients of the k active algebraic constraints).
When provided, each Newton step δX is projected onto the null space of the
active constraint gradients before being applied:
δX ← δX − Cᵀ(CCᵀ)⁻¹ C · δX.
When `None`, existing behavior is preserved exactly.

**`ConstraintAwareController`** (introduced in F4).  Wraps an existing
step-size controller (`PIController` or `VODEController`) and adds
constraint lifecycle management between accepted steps:
- evaluates |r⁺ⱼ − r⁻ⱼ| / max(r⁺ⱼ, r⁻ⱼ) per reaction pair;
- activates a constraint when the ratio falls below ε_activate and
  deactivates when it rises above ε_deactivate (hysteresis prevents
  chattering);
- applies consistent initialization — projects the state onto the
  newly-activated constraint manifold — before the next step;
- calls `project_conserved` after each accepted step;
- detects the NSE limit (rank of active constraint set equals
  n_species − n_conserved) and switches to a direct Newton solve on
  the n_conserved-dimensional conservation-law system.

---

#### Problem ladder

Each phase introduces one infrastructure piece, tests it on a synthetic
toy problem, and exercises the growing stack on a harder physics problem.
All tests register in `tests/test_time_integrators.py`.

| Phase | Physics problem | Infrastructure introduced | Synthetic tests |
|---|---|---|---|
| F1 ✓ | n-species decay chain (Aₙ → Aₙ₊₁, linear; `BlackBoxRHS`) | `ReactionNetworkRHS` protocol; stoichiometry analysis; conservation law derivation | 2-species A⇌B toy: verify S, conservation_basis = left null space of S, factored form __call__ = S·(r⁺−r⁻), detailed balance at equilibrium |
| F2 ✓ | Two-body fusion A + A → B (quadratic; `BlackBoxRHS`) | `project_conserved` | 3-species toy: orthogonal projection onto Σxᵢ = 1; idempotence; minimum-norm property; round-trip error ≤ ε_machine |
| F3 ✓ | Robertson problem (k₁=0.04, k₂=3×10⁷, k₃=10⁴; `JacobianRHS`) | Projected Newton iteration | 2D system with one hard algebraic constraint: Newton steps stay on constraint manifold; result agrees with exact reduced 1D Newton to integration tolerance |
| F4 ✓ | 5-isotope α-chain at fixed T (`ReactionNetworkRHS`) | Constraint activation state in `ODEState`; `ConstraintAwareController` | A⇌B toy: constraint activates when r⁺/r⁻→1; consistent initialization lands on manifold; hysteresis prevents chattering; deactivation restores ODE trajectory |
| F5 ✓ | 3-species A⇌B⇌C symmetric network (`ReactionNetworkRHS`) | `nonlinear_solve` in `_newton.py`; `solve_nse` in `constraint_aware.py`; NSE limit detection and direct NSE solve in `ConstraintAwareController`; absent-species rate-threshold guard in `_equilibrium_ratios` | A⇌B⇌C toy: both constraints activate simultaneously, `solve_nse` recovers A=B=C=1/3 to machine precision; 11-species hub-and-spoke: fast and slow spoke groups activate at distinct times (staggered activation), `nse_events` logged at full NSE, final Aᵢ=1/11; rate-threshold guard prevents spurious activation of absent-species pairs in chain topology |

#### Invariants upheld by this layer

- **Conservation is a hard pass criterion.**  Any integrator or controller
  that violates conservation beyond floating-point precision is a defect,
  not a known limitation.  `project_conserved` enforces this after every
  accepted step.
- **Constraint chattering is prevented by hysteresis.**  ε_activate = 0.01,
  ε_deactivate = 0.10 (10× ratio) was sufficient for all F4–F5 test
  problems.  Widen the gap if chattering is observed on non-monotone problems.
- **Dense Newton is O(n³).**  Acceptable for n ≤ O(100); sparse factorization
  belongs to Epoch 9 when production-scale networks arrive.

---

## Cross-cutting open questions

Items that are not scoped to any specific epoch.  They surface here so they
are not lost; the decision of when and how to schedule them is made when
the implementation lane becomes clear.

**`set_default_backend` vs. solver-level override (Epoch 4 carry-over).**
Time-integrator code currently inherits the process-wide default backend set
by `set_default_backend`.  If per-`Integrator` backend overrides are needed
(e.g., a JAX backend for one integrator while the rest use NumPy), a
keyword argument on `Integrator.__init__` is the natural extension point.
Defer until a concrete use case requires it.

**AMR integration state (Epoch 12 forward).**  The time-stepper must accept
hierarchical state once meshes refine.  Integrator state types (`ODEState`,
`NordsieckHistory`) may need coarse–fine variants that carry per-level
sub-states.  Defer until the AMR mesh hierarchy is defined.

**GR integrator coupling (Epoch 13 forward).**  Integrator state and step
program become coupled to a dynamical 3-metric; the structured-RHS protocol
may need a `WithMetricRHS` variant.  Defer until the GR lane begins.

**Temporal convergence across spatial dimensions.**  The convergence
framework verifies time-integration order on scalar and small-vector problems.
Once spatial dimensions are coupled (Epoch 5+), temporal claims should be
re-verified on 2D and 3D problems.

**HDF5 checkpoint/restart.** Persistent serialization of simulation state
(time-stepper state, mesh fields, parameters) with provenance sidecars
(git hash, timestamp, parameter record).  GPU-written checkpoints must be
readable on CPU-only machines.  Not assigned to an epoch; the right time
to schedule depends on which integrator state types end up carrying
non-serializable data and on whether checkpointing arrives as part of a
broader persistence layer or as a standalone capability.

---

## Current work

### Sprint: Architecture ownership structural claims

Goal: make architecture ownership a reusable, machine-checkable capability in
the test suite.  Numerical claims verify that an algorithm is correct on a
problem.  Architecture ownership claims verify that the codebase has one
intended owner for a responsibility, that public exports are deliberately
categorized, and that helper/policy objects do not quietly become competing
implementations.  The validation home for this capability is
`tests/test_structure.py`; feature-specific tests may remain in their existing
files, but ownership maps and anti-duplication checks belong in the structural
claim registry.

The first application is the time-integration layer.  The recent Nordsieck
cleanup made `VODEController` the single public adaptive Nordsieck controller,
with `OrderSelector` and `StiffnessSwitcher` retained as reusable policies.
This sprint should encode that decision structurally so the wrappers cannot
reappear accidentally, and so future packages can reuse the same pattern.

The sprint is complete when the following are true:

- **Reusable ownership claim type.**  `tests/test_structure.py` exposes a
  declarative claim for package-level architecture ownership.  A claim can
  categorize public symbols, name exclusive responsibility owners, list
  forbidden public symbols, and fail with diagnostics that identify the
  mismatched symbol or responsibility.
- **Public-category maps.**  Ownership claims can assert that each public export
  in a package belongs to an expected category such as integrator, controller,
  policy, RHS wrapper, decomposition, solver, result object, or helper.  The
  categories are deliberately local to each package so the mechanism is general
  without imposing a global taxonomy too early.
- **Exclusive-owner checks.**  Ownership claims can assert that a named
  responsibility has exactly one public owner.  This is the guard against
  parallel implementations at the same abstraction level: if a new class claims
  the same role, the structural test fails until the architecture map is
  consciously updated or the duplication is removed.
- **Forbidden-symbol checks.**  Ownership claims can assert that retired public
  names remain absent.  This catches accidental reintroduction of wrappers or
  compatibility aliases when the project has explicitly consolidated a
  responsibility.
- **Time-integrator ownership map.**  The first concrete map covers
  `cosmic_foundry.computation.time_integrators`.  It should classify public
  method families, drivers/controllers, policies, RHS wrappers, domains,
  coefficient/history objects, and helpers.  It should encode at least:
  `VODEController` owns adaptive Nordsieck control; `Integrator` owns the
  generic integrator/controller advance loop; `ConstraintAwareController` owns
  reaction-network constraint lifecycle advancement; `OrderSelector` and
  `StiffnessSwitcher` are policies, not competing controllers.
- **Time-integrator anti-duplication guard.**  The time-integrator map forbids
  retired wrapper names including `VariableOrderNordsieckIntegrator` and
  `FamilySwitchingNordsieckIntegrator`.  If the first implementation discovers
  other same-level overlaps or ambiguous ownership, fix the code or update the
  map in the same PR rather than documenting the ambiguity as acceptable.
- **Generalization path.**  The structure-claim machinery is documented in code
  well enough for later PRs to add maps for solvers, decompositions, discrete
  operators, geometry, and autotuning without copying test logic.

Recommended PR sequence:

1. Add the reusable ownership-claim machinery to `tests/test_structure.py` and
   implement the time-integrator ownership map.  Fix any time-integrator
   ownership gaps or overlaps the claim exposes.
2. Add ownership maps for linear solvers and decompositions, reusing the same
   claim machinery.
3. Add ownership maps for discrete operators and geometry/theory boundaries,
   reusing the same claim machinery.

---

## Physics roadmap

### Foundation epochs

| Epoch | Layer | Capability |
|-------|-------|------------|
| 0 | Theory / Geometry | **Mathematical foundations. ✓** Layer architecture and symbolic-reasoning import boundary; `foundation/`, `continuous/`, `discrete/`, `geometry/` type hierarchies; `CellComplex`, `Mesh`, `StructuredMesh`, `DiscreteField`, `VolumeField`, `RestrictionOperator`; process discipline M0–M2. |
| 1 | Geometry / Validation | **Observational grounding. ✓** `EuclideanManifold`, `CartesianChart`, `CartesianMesh`; first `validation/` notebook (Schwarzschild spacetime, GPS time dilation); settles `SymbolicFunction` interface and `Point` type (M3). |
| 2 | Discrete | **FVM Poisson solver. ✓** `PoissonEquation`; `DiffusiveFlux(2,4)`; `DivergenceFormDiscretization` + `NumericalFlux` family; oracle-free convergence framework; SPD analysis; `LinearSolver` ABC with `DenseJacobiSolver` and `DenseLUSolver`; end-to-end O(hᵖ) convergence sweep. FVM machinery reused from Epoch 6 onward. |
| 3 | Computation | **Backend-agnostic computation layer. ✓** `Tensor` (arbitrary rank, `Real` protocol); `Backend` protocol with `PythonBackend`, `NumpyBackend`, `JaxBackend`; mixed-backend arithmetic guards; AST-based numeric-import boundary; self-calibrating roofline performance gate; `LazyDiscreteField` collapsed into `FaceField` and `_BasisField`. |
| 4 | Computation | **Time integration layer. ✓** Six-axis DSL (RHS protocol, state, step program, coefficient algebra, controller, verification primitives) with explicit RK as the first instantiation; phases extend to adaptive control, B-series verification, symplectic, implicit, IMEX, multistep, variable-order, exponential, and splitting families; reaction-network RHS with stoichiometry analysis, constraint lifecycle management, and NSE limit detection via `solve_nse`. |

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
