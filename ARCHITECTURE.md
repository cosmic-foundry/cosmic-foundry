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
├── AdditiveRHS                  — (explicit, implicit) split for IMEX
├── HamiltonianSplit             — (dH_dq, dH_dp) for symplectic methods
├── LinearPlusNonlinearRHS       — (L, N) split for exponential integrators
└── OperatorSplitRHS             — [f_1, …, f_k] for operator splitting
                                   (SplittingStep sequence drives substep weights)

State types:

RKState(NamedTuple)              — (t, u, dt, err); used by RK, exponential,
                                   IMEX, implicit, and splitting integrators
MultistepState(NamedTuple)       — (t, u, dt, err, history); used by explicit
                                   multistep integrators
NordsieckState                   — Nordsieck history vector for multistep methods

Integrators:

RungeKuttaIntegrator             — Butcher-tableau explicit RK (arbitrary order)
                                   instances: forward_euler(1), midpoint(2), heun(2),
                                   ralston(2), rk4(4), bogacki_shampine(3,embedded),
                                   dormand_prince(5,embedded)
DIRKIntegrator                   — diagonally implicit RK
                                   instances: backward_euler(1), implicit_midpoint(2),
                                   crouzeix_3(3)
IMEXIntegrator                   — additive RK (paired explicit + implicit tableaux)
                                   instances: ars222(2)
ExplicitMultistepIntegrator      — explicit linear multistep (Adams-Bashforth)
                                   instances: ab2, ab3, ab4
NordsieckIntegrator              — fixed-order Nordsieck-form BDF / Adams-Moulton
                                   factories: bdf_family → bdf1–bdf4
                                              adams_family → adams_moulton1–adams_moulton4
VariableOrderNordsieckIntegrator — online order selection (OrderSelector)
FamilySwitchingNordsieckIntegrator
                                 — runtime BDF ↔ Adams-Moulton switching (StiffnessSwitcher)
ExponentialEulerIntegrator       — ETD-Euler, order 1; instance: etd_euler
ETDRK2Integrator                 — order 2; instance: etdrk2
CoxMatthewsETDRK4Integrator      — order 4 (classical); instance: cox_matthews_etdrk4
KrogstadETDRK4Integrator         — order 4 (stiff-order-correct); instance: krogstad_etdrk4
SymplecticCompositionIntegrator  — position-Verlet family for separable Hamiltonian
                                   systems; inherits TimeIntegrator; accepts
                                   HamiltonianSplit with split_index
                                   instances: symplectic_euler(1), leapfrog(2),
                                   forest_ruth(4), yoshida_6(6), yoshida_8(8)
StrangSplittingIntegrator        — meta-integrator composing sub-integrators;
                                   factories: lie_steps()(1), strang_steps()(2),
                                   yoshida_steps()(4, negative substep weights)

Controllers:

ConstantStep                     — fixed step size
PIController                     — Gustafsson PI formula with accept/reject
VODEController                   — VODE-style Nordsieck-aware step control

Infrastructure:

TimeStepper                      — drives integrator + controller loop;
                                   advance(rhs, u0, t0, t_end) → RKState
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

## Current work: Epoch 4 — Time integration layer

### Unification roadmap

The current layer works but contains two forms of fragmentation that
compound as the integrator family grows.  The roadmap below resolves
both, producing a single coherent entry point whose algorithm selection
is driven entirely by the mathematical structure of the problem.

---

#### Vocabulary — mathematically named replacements

The table below replaces every name rooted in a specific algorithm or
implementer convention with a name that describes mathematical function.
The right column is the target name; the left column is the current
name.  The migration is breaking; it happens in phases (see below).

| Current name | Target name | Rationale |
|---|---|---|
| `RKState` | `ODEState` | A Runge-Kutta state is just an ODE integration state: (t, u, dt, err). The name should not imply method family. |
| `NordsieckState` | `MultistepState` | The Nordsieck encoding is an implementation detail; the concept is a multistep history of past solution data. |
| `PartitionedState` | *(eliminated — done)* | Folded into `RKState.u = concat([q, p])`; `SymplecticCompositionIntegrator` unpacks via `HamiltonianSplit.split_index`. |
| `DIRKIntegrator` | `ImplicitRungeKuttaIntegrator` | DIRK is one implementation strategy for implicit RK; the public name should say "implicit Runge-Kutta". |
| `IMEXIntegrator` | `AdditiveRungeKuttaIntegrator` | IMEX is the physics abbreviation; the mathematical name is additive Runge-Kutta (ARK). |
| `NordsieckIntegrator` | `MultistepIntegrator` | Same reasoning as `NordsieckState`. |
| `StrangSplittingIntegrator` | `CompositionIntegrator` | Strang is one composition scheme; the class is the general composition meta-integrator. |
| `SymplecticSplittingIntegrator` | `SymplecticCompositionIntegrator` *(done)* | Specializes `CompositionIntegrator` with symplecticity constraint. |
| `LinearPlusNonlinearRHS` | `SemilinearRHS` | du/dt = Lu + N(t,u) is a semilinear ODE; the name should say so. |
| `AdditiveRHS` | `SplitRHS` | Carries an explicit/implicit split; "additive" is the ARK term, "split" is the mathematical concept. |
| `HamiltonianSplit` | `HamiltonianRHS` | The protocol describes a Hamiltonian RHS; "split" implies an algorithmic choice. |
| `OperatorSplitRHS` | `CompositeRHS` | Carries k component RHS objects; "composite" names the mathematical structure. |

---

#### Axis A — unified dispatch surface

**Goal.** A single `Integrator.step(rhs, state, dt)` entry point that
selects the algorithm family by inspecting `rhs` against the protocol
hierarchy.  No caller changes the integrator when they change the RHS;
the dispatch is transparent.

**Design.**

```
Integrator (ABC / Protocol)
  step(rhs: RHSProtocol, state: ODEState, dt: float) -> ODEState
```

The concrete `AutoIntegrator` implements `step` as a dispatch chain:

```python
if isinstance(rhs, SemilinearRHS):
    # exponential integrator family
elif isinstance(rhs, HamiltonianRHS):
    # symplectic composition family
elif isinstance(rhs, CompositeRHS):
    # general composition family
elif isinstance(rhs, SplitRHS):
    # additive RK (ARK) family
elif isinstance(rhs, ImplicitRHS):   # WithJacobianRHSProtocol
    # implicit RK family
else:
    # explicit RK family (default)
```

The dispatch chain is a total function: every `RHSProtocol` falls into
exactly one branch.  Adding a new RHS type adds a new branch; existing
callers are unaffected.

**Existing specialist integrators** (`RungeKuttaIntegrator`,
`ImplicitRungeKuttaIntegrator`, …) remain as first-class objects.
`AutoIntegrator` is a convenience wrapper, not a replacement; users who
know their algorithm keep using the specific class.

**Type coherence requirement.** All specialist integrators must satisfy
the `Integrator` protocol — concretely, `DIRKIntegrator` and
`IMEXIntegrator` must inherit (or structurally match) the same
`TimeIntegrator` ABC that `RungeKuttaIntegrator` currently inherits.
This is the most impactful single cleanup and unblocks Axis A.

---

#### Axis B — unified state type

**Goal.** Replace four incompatible state types with one.  The
`ODEState` type carries optional slots for structure that only some
methods use; methods that do not need a slot ignore it.

**Target definition.**

```
ODEState:
    t:    float          — current time
    u:    Tensor         — current solution vector (or structured pair for Hamiltonian)
    dt:   float          — last accepted step size
    err:  float          — last local error estimate (0.0 if not available)
    history: MultistepHistory | None
               — ordered ring buffer of past (t_k, u_k, f_k) tuples;
                 None for single-step methods
```

**Migration path per affected type.**

- `NordsieckState` → `ODEState` with `history = MultistepHistory(nordsieck_vector)`.
  `MultistepIntegrator` reads/writes `history`; single-step integrators
  pass through `state.history = None`.

- `MultistepState(t, u, dt, err, history)` → `ODEState` with `history =
  MultistepHistory(history_list)`.  `ExplicitMultistepIntegrator` stores
  past `f` evaluations in `history.derivatives`; the Nordsieck form stores
  scaled derivatives directly — the buffer layout is method-specific but
  the type is shared.

**Breaking change surface.** `TimeStepper.advance` return type changes
from `RKState` to `ODEState`; field names are identical (`t`, `u`, `dt`,
`err`), so destructuring callsites are unaffected.  Callers that type-
annotate the return must update the annotation.

---

#### Phased implementation plan

Each phase is a self-contained PR that leaves all tests green and adds
at least one new test for the structural change.

| Phase | Title | Scope |
|---|---|---|
| D | **Unified state** | Rename `RKState → ODEState`; merge `NordsieckState` into `ODEState` with `history` slot; `TimeStepper.advance` returns `ODEState`. |
| E | **Rename sweep** | All target names from the vocabulary table replace current names; deprecation warnings on old names for one release cycle; B-series verification re-exports under new names. |
| F | **`AutoIntegrator`** | Implement dispatch chain; add integration test that passes each RHS type through `AutoIntegrator` and verifies correct order. |

Phase D can proceed immediately.  Phases E and F require D.

---

**Open questions:**

1. **`set_default_backend` vs. solver-level override.** Time-integrator
   code must inherit whichever resolution lands; a per-`TimeStepper`
   backend override is the natural API extension if process-wide defaults
   turn out to be insufficient.

**Design decisions to revisit in later epochs:**

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
