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
AdaptiveNordsieckController                   — adaptive Nordsieck controller combining
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

IntegrationDriver               — drives integrator + controller loop;
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
step-size controller (`PIController` or `AdaptiveNordsieckController`) and adds
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
by `set_default_backend`.  If per-`IntegrationDriver` backend overrides are
needed (e.g., a JAX backend for one integrator while the rest use NumPy), a
keyword argument on `IntegrationDriver.__init__` is the natural extension point.
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

### Sprint: Parameter-space algorithm ownership atlas

Goal: make the problem parameter space the primary artifact, then derive
algorithm ownership, selection, generated documentation, and coverage tests from
how implementations cover that space.  The previous capability maps made
ownership executable, but they still start from implementations and attach
qualitative tags such as `symmetric_positive_definite`, `rank_deficient`, or
`domain_aware_acceptance`.  This sprint should invert that hierarchy: first
declare the axes of the problem space, then let algorithms claim bounded regions
inside that space using numeric predicates, certificates with stated tolerances,
and cost estimates.

The first application is the linear solver and decomposition stack because its
ownership range has the least subjective mathematical vocabulary.  Even there,
the atlas should not start with a prose `problem_kind = linear_system` axis.
The underlying mathematical object is a solve relation: find an unknown
`x ∈ X` such that a residual/objective/constraint relation involving a map
`R : X -> Y` satisfies an acceptance predicate.  Named classes such as "linear
system", "least squares", "nonlinear root", and "eigenproblem" are derived
regions over axes describing `X`, `Y`, `R`, available oracles, targets,
constraints, and acceptance semantics.

A solver should not merely claim to own "SPD systems"; it should declare a
predicate over primitive descriptor coordinates such as:

```
symmetry_defect(A) <= eps_sym
coercivity_lower_bound(A) >= alpha_min
condition_estimate(A) <= kappa_max
rhs_consistency_defect(A, b) <= eps_rhs
predicted_memory_bytes(A, b) <= budget.memory_bytes
predicted_work_fmas(A, b, tol) <= budget.work_fmas
```

Qualitative vocabulary remains useful only as a compact label derived from the
space.  For example, `symmetric_positive_definite` is acceptable only if it is an
alias for a region defined by bounded `symmetry_defect` and
`coercivity_lower_bound` axes.  The validation home for these architecture
claims remains `tests/test_structure.py`: the claim registry should hold the
parameter-space schemas, coverage patches, derived aliases, invalid-cell rules,
and gap annotations that feed both tests and documentation.  Numerical test
modules should own the descriptor estimators and problem fixtures needed to
check the mathematics.

The same schema should generate capability coverage documentation.  A developer
should be able to open one generated page and see the gridded parameter space
with owned, explicitly rejected, and uncovered cells.  Explicit missing regions
are useful after a gap is noticed; the grid is what exposes gaps that nobody has
named yet.

This is not just a cleaner naming scheme.  The meta-level goal is to make
algorithm ownership an executable epistemic model: separate the mathematical
space being described, the evidence available for locating a concrete problem in
that space, the regions our implementations claim, and the projection used to
explain those claims to humans.  Prose names such as "linear solver" or "SPD"
are allowed only as views over that model.  They must not become independent
sources of truth.

The sprint is complete when the following are true:

- **Parameter-space schema is primary.**  Each package that participates in
  capability selection declares a `ParameterSpaceSchema`: named axes, finite bins
  or numeric intervals for those axes, allowed cross-axis combinations, units or
  norm definitions, and the descriptor fields needed to place a concrete problem
  into the space.  Capabilities are not allowed to introduce private axes; a new
  ownership claim first extends the schema, then declares coverage over it.
- **Descriptors are points or cells.**  `AlgorithmRequest` and
  `AlgorithmCapability` can be evaluated against a typed descriptor that locates
  a concrete problem in the schema.  A descriptor may be a point
  (`n = 128`, `symmetry_defect = 2e-14`) or a conservative cell
  (`conditioning = unknown_condition`).  Unknown descriptor values are explicit:
  selection must either reject the request, choose a capability that can certify
  the value internally, or require an explicit fallback policy.
- **Capabilities are coverage patches.**  A capability declares a bounded region
  in the parameter space plus the guarantees it provides inside that region.
  The region may be a conjunction of bins, numeric intervals, and cost-model
  inequalities over schema axes.  The implementation name is metadata attached
  to the region, not the root of the model.
- **Selection is a point query.**  Runtime dispatch becomes a query of the
  parameter-space atlas: locate the descriptor cell, find all coverage patches
  containing it, reject if none own it, and fail on overlap unless priority data
  is explicit.  This keeps dispatch, generated documentation, and structure
  tests as different views of the same object.
- **Knowledge state is modeled separately from truth.**  A coordinate such as
  "symmetric", "full rank", or "well conditioned" has two parts: the underlying
  mathematical predicate and the evidence currently available to the selector.
  Exact facts, certified bounds, estimates, caller assumptions, and unknowns are
  distinct states.  A capability owns the request only when its contract says
  which evidence states it accepts and how uncertainty affects the bound.
- **Bound vocabulary is finite and inspectable.**  Bounds are structured data,
  not ad hoc lambdas or prose.  The initial operators should include only
  comparison predicates over named descriptor fields, membership predicates for
  finite sets, and simple affine/cost comparisons.  If a contract needs a new
  predicate kind, the PR adding it must also add structural tests for that kind.
- **Solve-relation descriptor.**  Add a descriptor for solve requests whose
  primitive coordinates can be measured, estimated, or certified: unknown-space
  dimension `dim_x`; residual/target-space dimension `dim_y`; number of scalar
  auxiliary unknowns; number of equality constraints; number of normalization
  constraints; residual target availability; map linearity defect; matrix
  representation availability; operator-application availability; derivative
  oracle kind (`none`, `matrix`, `jvp`, `vjp`, `jacobian_callback`);
  objective relation (`none`, `residual_norm`, `least_squares`, `spectral_residual`);
  acceptance relation (`residual_below_tolerance`, `objective_minimum`,
  `stationary_point`, `eigenpair_residual`); requested residual tolerance;
  requested solution tolerance; backend kind; device kind; and work/memory
  budgets.
- **Linear-operator sub-descriptor.**  When the solve relation contains a linear
  map or a linearized derivative, add numeric coordinates for matrix
  availability; assembly cost; matvec cost; memory estimate; symmetry defect;
  skew-symmetry defect; diagonal nonzero margin; diagonal dominance margin;
  coercivity lower bound; singular-value lower bound; condition estimate; rank
  estimate; nullity estimate; and RHS consistency defect.  Derived categories
  such as square, overdetermined, full rank, rank deficient, SPD, and
  matrix-free are aliases over these primitive coordinates.
- **Linear solver schema.**  The solver parameter space should start with
  orthogonal axes for solve-relation structure rather than named problem kinds:
  `dim_x`, `dim_y`, auxiliary scalar count, residual map linearity defect,
  target availability, objective relation, acceptance relation, constraint
  counts, derivative/oracle availability, operator representation, rank/nullity
  estimates, symmetry/skew-symmetry defects, coercivity lower bound, condition
  estimate, RHS consistency defect, and resource budgets.  Axes are declared
  independently of current implementations so an empty cell is a real absence,
  not an omitted example.  Named regions are derived:

  ```
  linear system:
    map_linearity_defect <= eps
    dim_x == dim_y
    residual_target_available = true
    acceptance_relation = residual_below_tolerance

  least squares:
    map_linearity_defect <= eps
    objective_relation = least_squares
    residual_target_available = true

  nonlinear root:
    map_linearity_defect > eps or unknown
    residual_target_available = false or target_is_zero = true
    acceptance_relation = residual_below_tolerance

  eigenproblem:
    auxiliary_scalar_count >= 1
    normalization_constraint_count >= 1
    acceptance_relation = eigenpair_residual
  ```
- **Schema validity is explicit.**  The atlas records which axis products are
  meaningful and which are invalid.  For example, symmetry and coercivity axes
  apply to linear maps or derivative maps, not to an arbitrary residual relation
  without a chosen linearization; SPD/coercivity ownership requires
  `dim_x == dim_y`; eigenpair regions require a normalization constraint and an
  auxiliary spectral parameter.  Invalid cells are rendered separately from
  uncovered-but-valid cells so the documentation distinguishes impossible
  combinations from missing algorithms.
- **Norm definitions are fixed.**  Descriptor fields that use norms must name
  the norm and scaling.  The default matrix defect is relative Frobenius norm:
  `||A - A.T||_F / max(||A||_F, eps)`.  The default residual defect is
  `||b - A x||_2 / max(||b||_2, eps)`.  Any different norm must be encoded in
  the descriptor field name or metadata rather than implied by prose.
- **Certificate source is explicit.**  Every descriptor field records whether it
  is exact, estimated, bounded from above, bounded from below, assumed by the
  caller, or unavailable.  Selection may trust exact and certified bounds;
  caller assumptions are allowed only when the request explicitly permits them,
  and tests must cover that policy.
- **Solver coverage patches.**  Linear solver capabilities are coverage patches
  over the schema.  Examples: `DenseCGSolver` covers descriptors whose residual
  map is certified linear, `dim_x == dim_y`, symmetry defect is below tolerance,
  coercivity lower bound is positive, condition estimate and iteration cost fit
  budget, and a matrix-free or assembled matvec is available.  `DenseJacobiSolver`
  covers the diagonally dominant stationary-iteration region.  `DenseLUSolver`
  covers full-rank square dense linear maps inside memory/work budget.
  `DenseSVDSolver` covers rank-deficient or minimum-norm dense regions.
  `DenseGMRESSolver` covers nonsymmetric matrix-free linear maps only under
  restart, memory, and predicted-work bounds.
- **Decomposition coverage patches.**  Decomposition capabilities are coverage
  patches over the dense-matrix subspace.  LU covers full-rank square dense
  matrices within cost budget; SVD covers rank-deficient, ill-conditioned,
  least-squares, or minimum-norm dense regions within factorization budget.
- **Cost models are contracts too.**  Capabilities declare a conservative
  symbolic cost model in terms of descriptor fields.  The first version can use
  coarse asymptotic coefficients such as dense `O(n^3)` factorization, dense
  `O(n^2)` solve, and iterative `iterations * matvec_cost`, but it must produce
  comparable numeric work and memory estimates for selection.
- **Coverage atlas generation.**  `tests/test_structure.py` is the source of
  truth for coverage documentation.  Parameter-space schemas and capability
  coverage patches are declared as structural claims or claim inputs there, and
  a generator renders the documentation page from that same data.  The page
  projects the space onto readable axis pairs or small tables, then overlays
  selected owners, intentional rejections, invalid cells, and
  uncovered-but-valid cells.  It must include the predicate bounds, the
  certificate sources accepted by the selector, the cost model, and the priority
  rule for every owned overlap.  It must also overlay numerical-test coverage:
  which descriptor cells have correctness, convergence, performance, or
  regression claims exercising the owned capability, which claim file owns that
  evidence, and whether the evidence is representative sampling or boundary
  sampling.  Uncovered cells remain visible even before anyone has written a
  missing-capability note for them.
- **Coverage projections are honest.**  A rendered atlas page is a projection of
  a higher-dimensional schema, not the schema itself.  Each plot or table must
  state which axes are shown, which axes are fixed, which axes are marginalized
  into a summary marker, and whether any hidden-axis overlap or gap exists.  A
  cell may not be rendered as simply "owned" when ownership depends on an
  unshown certificate, budget, tolerance, or backend axis.
- **Gaps are first-class regions.**  A missing algorithm can be represented as an
  explicitly named unowned descriptor region after the coverage atlas exposes
  it.  For example, the current solver coverage page should show blank coverage
  over the derived nonlinear-root region; that blank can then be
  tagged as:

  ```
  Region: nonlinear algebraic solve F(x) = 0
  Descriptor:
    map_linearity_defect > eps or unknown
    residual_target_available = false or target_is_zero = true
    jacobian_available in {true, false}
    acceptance_relation = residual_below_tolerance
    requested_residual_tolerance = finite
  Selected owner: none
  Existing partial owners:
    time_integrators._newton.nonlinear_solve is internal stage machinery, not a
    public nonlinear-system solver capability.
  Required capability before this region is owned:
    NonlinearSolver with descriptor bounds for residual norm, Jacobian
    availability, local convergence radius or globalization policy, line-search
    or trust-region safeguards, max residual evaluations, and failure reporting.
  ```

  The important point is that the nonlinear-root region is derived from
  primitive axes before this note is written.  The note records a known gap; the
  axis grid is what gives us a chance to find unknown gaps.
- **Coverage is tested at the schema level.**  Structural tests validate the
  schema before any algorithm selection: every axis has bins or intervals; every
  descriptor field referenced by an axis exists; every coverage patch references
  declared axes; every invalid cell has a reason; every generated documentation
  cell comes from the schema; and the rendered coverage document is byte-for-byte
  current with the claim registry.  Then tests sample representative cells: SPD
  well-conditioned dense systems select CG or LU only when priority data says
  why; rank-deficient minimum-norm requests select SVD; strictly diagonally
  dominant systems select Jacobi when the iteration budget is satisfied;
  nonsymmetric matrix-free systems select GMRES; unsupported descriptors reject.
  Numerical claim registries should expose enough descriptor metadata for the
  atlas to overlay tested cells on top of owned cells.  An owned region without
  numerical evidence is allowed only when it is explicitly marked structural-only
  or deferred, so the documentation can distinguish "claimed and tested" from
  "claimed but not numerically exercised".  Ambiguous overlap without explicit
  priority remains a test failure.
- **No prose-only ownership.**  The generated documentation may contain human
  explanation, but every ownership, rejection, invalidity, overlap, priority, and
  missing-region statement in it must trace back to a structural claim in
  `tests/test_structure.py`.  Conversely, every structural claim that changes
  dispatch behavior must appear in the generated atlas or in an explicit
  machine-readable exclusion list.  This prevents the documentation from
  becoming a parallel taxonomy that can drift from the selector.
- **Qualitative tags become schema aliases.**  Existing string-set capability
  tags may remain temporarily, but the sprint should move high-value tags onto
  schema aliases.  An alias such as `full_rank` must expand to a rank or
  singular-value interval; `matrix_free` must expand to operator-representation
  and assembly-cost axes; `linear_system`, `least_squares`, `nonlinear_root`,
  and `eigenproblem` must expand to solve-relation regions; and
  `domain_aware_acceptance` must later expand to domain-distance and
  retry-policy axes.
- **Generalization path.**  After the linear solver/decomposition version is in
  place, time integrators should receive descriptors for domain distance,
  stiffness estimates, Jacobian availability, local error target, step retry
  budget, and RHS evaluation cost.  Discrete operators should receive descriptors
  for mesh regularity, geometry type, stencil width, formal order, boundary
  closure, conservation form, and smoothness assumptions.  These later maps reuse
  the same bound and descriptor machinery rather than introducing package-local
  predicate systems.

Recommended PR sequence:

1. Add the solve-relation, linear-solver, and decomposition parameter-space
   schemas before changing any solver ownership.  Tests should prove that named
   problem classes such as linear system, least squares, nonlinear root, and
   eigenproblem are derived regions over primitive axes, not primary axes, and
   that every generated coverage cell maps to a valid descriptor template.
2. Generate a capability coverage document from the schemas and coverage patches,
   all sourced from `tests/test_structure.py`, that visualizes owned, rejected,
   invalid, uncovered, and numerically tested cells.  The structural test should
   fail if the generated documentation is stale.  Seed it with the public
   nonlinear-solver gap as an annotation on an already-visible uncovered
   nonlinear-root region.
3. Add `LinearOperatorDescriptor` construction for small assembled operators and
   direct descriptor fixtures in `tests/test_structure.py`.  Keep estimation
   conservative and deterministic; do not use performance timing as a source of
   truth for ownership.
4. Convert linear solver capabilities to coverage patches and update
   selector tests for SPD, diagonally dominant, rank-deficient, nonsymmetric,
   matrix-free, over-budget, and unknown-descriptor cases.
5. Convert decomposition capabilities to coverage patches, including
   rank threshold, minimum-norm semantics, dense memory budget, and work budget.
6. Add a follow-up sprint plan for quantitative time-integrator descriptors once
   the solver/decomposition predicates have stabilized.

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
