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
physical models (PDE operators, discretization schemes) and houses `State`,
the concrete simulation-state type that couples mesh geometry with numeric
storage via `computation/`. `physics/` may import from all other packages.

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
| `TensorField`, `SymmetricTensorField` | **missing** | rank-(p,q) annotated discrete fields; needed Epoch 6+ (rotating-frame metric, MHD) |
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
                                  and CartesianRestrictionOperator (degree = ndim−1).
                                  Concrete subclass:
                                    _CallableFaceField — callable-backed (NumericalFlux,
                                                         CartesianRestrictionOperator)

RestrictionOperator(NumericFunction[F, DiscreteField[V]])
                               — free: mesh: Mesh;
                                  formal bridge from continuous/ to discrete/:
                                  a Function plus a Mesh yields a DiscreteField.
                                  F is a generic input type so that concrete
                                  subclasses can narrow it (e.g. SymbolicFunction)
                                  without an LSP violation.
                                  Degree selects the cochain level:
                                    degree=ndim:   ∫_Ωᵢ f dV → VolumeField  (n-cochain)
                                    degree=ndim-1: ∫_Aᵢ F·n̂ dA → FaceField
                                    degree=1:      ∫_eᵢ F·dl → EdgeField
                                    degree=0:      f(vᵢ) → PointField

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

DiscreteOperator(NumericFunction[DiscreteField, DiscreteField])
                            — the output of Discretization; the Lₕ that makes
                              Lₕ ∘ Rₕ ≈ Rₕ ∘ L hold to the chosen order.
                              Earns its class via two falsifiable claims:
                                order: int — composite convergence order
                                continuous_operator: DifferentialOperator —
                                  the continuous operator this approximates
                                  (threaded automatically by Discretization
                                  from its input L)
                              Not independently constructed from stencil
                              coefficients; produced by a Discretization.

NumericalFlux(DiscreteOperator)
                            — a DiscreteOperator with the cell-average →
                              face-flux calling convention:
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

CartesianRestrictionOperator(RestrictionOperator[SymbolicFunction, sympy.Expr])
                                       — Rₕᵏ: Ωᵏ → k-cochains via exact SymPy integration.
                                         degree=ndim:   Rₕⁿ(n-Form)   → VolumeField (∫_Ωᵢ f dV, total)
                                                        (ThreeForm in 3-D, OneForm in 1-D)
                                         degree=ndim-1: Rₕⁿ⁻¹(OneForm) → FaceField   (face-normal flux ∫ F_a dA_⊥)
                                                        OneForm used as Cartesian (n-1)-Form proxy:
                                                        F.component(a) stands for *(F)_a dA_⊥
                                         degree=1:      Rₕ¹(OneForm)   → EdgeField   (edge line integral)
                                         degree=0:      Rₕ⁰(ZeroForm)  → PointField  (vertex evaluation)
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
State(DiscreteField[float])     — concrete Tensor-backed simulation-state field.
                                  Stores cell-average values φ̄ᵢ = (1/|Ωᵢ|)∫_Ωᵢ f dV.
                                  Multi-index cell access via mesh shape.
                                  Backed by any Backend. The canonical type for
                                  time integrators, checkpoint/restart, and I/O.
                                  PythonBackend with sympy.Expr leaves also works,
                                  enabling symbolic evaluation for convergence proofs.

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

FVMDiscretization(Discretization)
                               — free: mesh, numerical_flux, boundary_condition
                                 Concrete FVM scheme; generic over
                                 DivergenceFormEquation. For each cell Ωᵢ,
                                 evaluates ∮_∂Ωᵢ F·n̂ dA by delegating to the
                                 NumericalFlux at each face; BC enters through
                                 boundary_condition.
                                 Not specialized to any particular conservation law:
                                 Epoch 2 supplies DiffusiveFlux for Poisson;
                                 Epoch 5 supplies HyperbolicFlux for Euler.
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
    JaxBackend      — JAX array; planned (Epoch 3, C5).

LinearSolver        — mesh-agnostic interface: solve(a: Tensor, b: Tensor) → Tensor.
                      Accepts an assembled N×N stiffness matrix and an N-vector
                      RHS; returns the solution vector. Assembly and index mapping
                      are the caller's responsibility, keeping computation/ free
                      of theory/discrete/ and physics/ dependencies.
                      SCOPE: linear operators only. Epoch 5 hydro (nonlinear
                      flux) requires a separate NonlinearSolver / Newton
                      iteration. LinearSolver is not the shared machinery
                      for Epoch 5; only FVMDiscretization and NumericalFlux
                      are reused across epochs.
                      Ships DenseJacobiSolver (weighted Jacobi, ω derived
                      from Gershgorin bound; works for both order=2 and
                      order=4 stencils) and DenseLUSolver (direct, in-place
                      LU with partial pivoting). Both operate on Tensor;
                      linear algebra hand-rolled, no LAPACK. Convergence
                      tests cap at N ≤ 32 in 2-D (≤ 1024 unknowns).
```

**Planned additions (Epoch 3):** `JaxBackend`; explicit time integrators
(`RungeKutta2`, `RungeKutta4`); HDF5 checkpoint/restart with provenance sidecars.

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

## Current work: Epoch 3 — Computation layer

**Target.** A fully-capable, backend-agnostic computation layer: `Tensor`
operating on any registered backend (Python, NumPy, JAX), with JAX enabling
GPU/TPU execution and JIT compilation of full solve loops, plus the explicit
time integration needed by Epoch 4 hydro.

**C1 — Pure-Python Tensor class. ✓** Arbitrary-rank numeric array backed by
nested lists. `Real` protocol. `einsum` general contraction. `__matmul__` fast
paths for dot, vecmat, matvec, matmul; `einsum` fallback for exotic ranks.
Multi-index `__getitem__`/`__setitem__`. Element-wise `*`, `/`. `copy()`,
`norm()`, `diag()`, `svd()` (one-sided Jacobi). Solvers and discretization
migrated to operate on `Tensor` throughout.

**C2 — Numeric import boundary. ✓** `scripts/ci/check_numeric_imports.py`
(AST-based) enforces that `math`, `numpy`, `scipy`, `jax`, `torch` appear
only under `computation/`. Wired into `.pre-commit-config.yaml`.

**C3 — Roofline performance regression gate. ✓**
`tests/test_performance.py`: session-scoped fixture measures the machine's
pure-Python FMA rate at startup; 8 claims assert each operation completes
within `EFFICIENCY_FACTOR = 8` of the roofline prediction. Self-calibrating.

**C4 — Backend protocol: PythonBackend + NumpyBackend. ✓** `Backend` protocol
in `computation/backends/`; `Tensor` accepts `backend=` at construction;
`Tensor.to(backend)` converts; `PythonBackend` wraps existing pure-Python
logic; `NumpyBackend(dtype=None)` uses NumPy with dtype inferred from input
by default. Mixed-backend arithmetic raises `ValueError`. 38 backend
correctness claims in `tests/test_tensor_backends.py`.

**C5 — JaxBackend. ✓** `JaxBackend` in `computation/backends/jax_backend.py`
satisfies the `Backend` protocol using JAX arrays.  JAX arrays are immutable;
`slice_set` uses `.at[idx].set(value)` and returns the updated array.
`Tensor.__setitem__` now routes all index types through `backend.slice_set`,
reassigning `self._data` with the result — this is the only interface change
needed to support both mutable (Python/NumPy) and immutable (JAX) backends.
Backend correctness claims in `test_tensor_backends.py` extended to cover
`JaxBackend` for all claim types (roundtrip, arithmetic, conversion, slice,
factory, mixed-backend).  **JIT scope decision**: `@jax.jit` is not applied
per `Backend` method call; the caller applies it at the solver or time-step
level.  Tracing a full solve loop requires all shape/rank branches to be
static, which is a separate refactor deferred to C8 when the time integrator
provides a natural JIT boundary.

**C6 — Backend parity and performance. ✓ (NumpyBackend)** Added
`_NumpyParityPerfClaim` and `_BackendSpeedupClaim` to `test_performance.py`.
`_NumpyParityPerfClaim` asserts NumpyBackend Tensor overhead ≤
`NUMPY_PARITY_FACTOR = 2` of raw `np.matmul`/`np.matvec` for N ∈ {8, 16, 32}.
`_BackendSpeedupClaim` asserts NumpyBackend is at least 10× faster than
PythonBackend for matmul (N ∈ {8, 16, 32}) and 5× for matvec (N ∈ {16, 32}),
catching regressions where NumPy is accidentally bypassed.  JaxBackend
performance claims (GPU vs CPU) deferred to C8, where `@jax.jit` provides
the natural JIT boundary that makes JAX competitive.

**C7 — Collapse LazyDiscreteField. ✓** `LazyDiscreteField` deleted.
`FaceField` covers all face-indexed fields; `_BasisField` (private to
`Discretization`) is a `DiscreteField` unit basis for `assemble()`.
Ghost-cell extension is handled by `DirichletGhostCells` and `PeriodicGhostCells`
(concrete `DiscreteBoundaryCondition` subclasses).  All convergence claims pass.

**C8 — Explicit time integrators.** `TimeIntegrator` ABC; `RungeKutta2` and
`RungeKutta4`. Backend-agnostic; operates on `State`-valued fields. Lane B
derivation: truncation error O(hᵖ), p = 2, 4, confirmed symbolically.

**C9 — HDF5 checkpoint/restart.** Write/read `State`-valued fields with
provenance sidecars (git hash, timestamp, parameter record). GPU-written
checkpoints readable on CPU-only machines.

**Open questions — Epoch 3 design points:**

1. **`set_default_backend` vs. solver-level override.** The current design
   sets a process-wide default. If two solvers in the same process need different
   backends (e.g., CPU for assembly, GPU for iteration), a per-solver backend
   argument to `LinearSolver.solve` may be needed. Deferred to C7.

**Open questions carried forward into Epoch 3:**

- AMR (Epoch 11): fixed-mesh `Discretization` will need hierarchical extension.
- GR (Epoch 12): face geometry is state-dependent when the 3-metric is a dynamical field.
- Multi-dimensional convergence sweep: currently 1-D only; 2-D/3-D deferred.

---

## Physics roadmap

### Foundation epochs

| Epoch | Layer | Capability |
|-------|-------|------------|
| 0 | Theory / Geometry | **Mathematical foundations. ✓** Layer architecture and symbolic-reasoning import boundary; `foundation/`, `continuous/`, `discrete/`, `geometry/` type hierarchies; `CellComplex`, `Mesh`, `StructuredMesh`, `DiscreteField`, `VolumeField`, `RestrictionOperator`; process discipline M0–M2. |
| 1 | Geometry / Validation | **Observational grounding. ✓** `EuclideanManifold`, `CartesianChart`, `CartesianMesh`; first `validation/` notebook (Schwarzschild spacetime, GPS time dilation); settles `SymbolicFunction` interface and `Point` type (M3). |
| 2 | Discrete | **FVM Poisson solver. ✓** `PoissonEquation`; `DiffusiveFlux(2,4)`; `FVMDiscretization` + `NumericalFlux` family; oracle-free convergence framework; SPD analysis; `LinearSolver` ABC with `DenseJacobiSolver` and `DenseLUSolver`; end-to-end O(hᵖ) convergence sweep. FVM machinery reused from Epoch 5 onward. |
| 3 | Computation | **Backend-agnostic computation layer.** `Tensor` (arbitrary rank, `Real` protocol); `Backend` protocol with `PythonBackend`, `NumpyBackend`, `JaxBackend`; JIT-compiled solve loop; `State` (concrete `DiscreteField`); `TimeIntegrator` (RK2/RK4); HDF5 checkpoint/restart. In progress. |

### Physics epochs

| Epoch | Capability |
|-------|------------|
| 4 | Scalar transport: linear advection and diffusion on a `CartesianMesh` via FVM. First end-to-end simulation; validates the full pipeline. |
| 5 | Newtonian hydrodynamics: Euler equations, FVM Godunov, PPM reconstruction, HLLC/HLLE Riemann solvers. |
| 6 | Rotating reference frames: `RotatingChart` in `geometry/`; formally principled approach via metric change (fictitious forces = Christoffel symbols of the rotating-frame metric, not source terms); co-designed with Epoch 5 hydro validation tests. |
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

### Per-epoch verification standard

Every epoch must satisfy this checklist before it is considered verified:

- Derivation document with SymPy checks for any new numerical scheme (Lanes B and C)
- At least one externally-grounded convergence test against an analytical solution
  or observational data (not an engine-generated golden file); where an analytical
  solution exists, the relevant `NumericFunction.symbolic` is declared so the
  check runs automatically
- Lane A/B/C classification stated in the PR description
