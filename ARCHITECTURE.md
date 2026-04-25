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
handled by the time integrator (Epoch 3), not by these objects. This separation
is preserved under the 3+1 ADM decomposition: in GR, covariant equations
`∇_μ F^μ = S` decompose to `∂ₜ(√γ U) + ∂ᵢ(√γ Fⁱ) = √γ S(α, β, γᵢⱼ, Kᵢⱼ)`
— still a spatial divergence operator with metric factors entering through the
`Chart` and curvature terms in `source`.

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
                              Epoch 2 supplies a DiffusiveFlux for Poisson;
                              Epoch 5 supplies a HyperbolicFlux for Euler.
                              Specializations belong in the NumericalFlux —
                              not in a new Discretization subclass per equation.
                              Note: LinearSolver is NOT part of the Epoch 5
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
                              Riemann solver consumes. Epoch 5 ships
                              HyperbolicFlux(2, HLLC) and HyperbolicFlux(4, HLLC)
                              as instances — not subclasses.

LinearSolver                — solves Lₕ u = f for a *linear* DiscreteOperator Lₕ.
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

**Boundary condition application (Option B, Epoch 2 decision).** `FVMDiscretization`
takes the `BoundaryCondition` as a constructor parameter; the resulting
`DiscreteOperator` is the discrete analog of `L` on the constrained function
space `{φ : Bφ = g}`. This keeps the commutation diagram a property of a single
operator, and lets the Epoch 7 multigrid ask the discretization for coarse
operators rather than asking the operator for its BC. Not committed long-term:
if time-dependent `g` arrives with Epoch 5 hydro (inflow/outflow BCs that change
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

The only layer that may import numeric libraries (`math`, `numpy`, `jax`,
etc.); all other layers are restricted to the Python standard library and
approved symbolic packages. Enforced by `scripts/ci/check_numeric_imports.py`.

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
                      no external dependencies.
    NumpyBackend(dtype=None)
                    — NumPy ndarray; dtype inferred from input by default
                      or fixed to an explicit numpy dtype; vectorized via
                      BLAS/LAPACK.
    JaxBackend      — JAX array; planned (Epoch 3, C5).
```

**Planned additions (Epoch 3):** `JaxBackend`; field storage (`MeshFunction`
backed by `Tensor`); explicit time integrators (`RungeKutta2`,
`RungeKutta4`); HDF5 checkpoint/restart with provenance sidecars.

### Cross-cutting

**Numerical transcription discipline.**
Physics capabilities sourced from reference tables (EOS polynomial fits,
reaction networks, opacity tables) need a discipline governing how
numeric tables are transcribed, verified, and updated independently of
the derivation-first lane policy. This decision is deferred to Epoch 8
(microphysics), when the first such capability lands.

**Kernel composition model.**
Realized as the `Backend` protocol in `computation/backends/`. `Tensor`
is the single public interface; `Backend` is the strategy that governs
storage and execution. Backends are per-instance; dispatch is
identity-checked (`is`), not type-checked, so user-defined backends
satisfying the protocol are first-class. Open questions: `@jax.jit`
tracing policy for `JaxBackend` (per-call JIT vs. solver-level JIT);
whether `set_default_backend` is sufficient for solver-level selection
or a solver-level override is needed.

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

## Finalized epochs

**Epoch 0 — Mathematical foundations.** Established the layer architecture
(`foundation/`, `continuous/`, `discrete/`, `geometry/`, `computation/`) and
the symbolic-reasoning import boundary. Delivered `Set`, `Function`,
`TopologicalManifold`, `Manifold`, `PseudoRiemannianManifold`,
`DifferentialForm`, `DivergenceFormEquation`, `CellComplex`, `Mesh`,
`StructuredMesh`, `MeshFunction`, and `RestrictionOperator`. Wired process
discipline (M0–M2): branch/PR standards, convergence-testing helpers, and the
`ARCHITECTURE.md`-as-living-document convention.

**Epoch 1 — Observational grounding.** Implemented `EuclideanManifold`,
`CartesianChart`, and `CartesianMesh` in `geometry/`; built the first
`validation/` notebook (Schwarzschild spacetime and GPS time dilation) that
runs end-to-end in CI. Settled the `SymbolicFunction` interface on concrete
fields, the coordinate-to-chart binding, and the `Point` type (M3).

**Epoch 2 — FVM Poisson solver.** Delivered `PoissonEquation`,
`DiffusiveFlux(order)` with symbolically derived stencil coefficients,
`FVMDiscretization`, `DiscreteOperator`, `NumericalFlux` ABCs,
oracle-free convergence testing via `RestrictionOperator.degree`, SPD analysis
of the assembled operator, `LinearSolver` ABC with `DenseJacobiSolver`
(weighted Jacobi, ω from Gershgorin bound) and `DenseLUSolver` (direct LU),
and end-to-end O(hᵖ) convergence for p = 2 and p = 4. The FVM machinery
(`FVMDiscretization`, `NumericalFlux`) is reused in every subsequent physics
epoch; `LinearSolver` is scoped to linear operators only.

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

**C4 — Backend protocol: PythonBackend + NumpyBackend.** `Backend` protocol
in `computation/backends/`; `Tensor` accepts `backend=` at construction;
`Tensor.to(backend)` converts; `PythonBackend` wraps existing pure-Python
logic; `NumpyBackend(dtype=None)` uses NumPy with dtype inferred from input
by default. Mixed-backend arithmetic raises `ValueError`. 38 backend
correctness claims in `tests/test_tensor_backends.py`.

**C5 — JaxBackend.** Implement `JaxBackend` satisfying the `Backend`
protocol using JAX arrays. Verify backend correctness claims pass. Settle
the `@jax.jit` tracing policy: whether JIT is applied per Backend method call
or at the solver level.

**C6 — Backend parity and performance.** Benchmark all three backends on
representative solver workloads. Establish that `NumpyBackend` is within 2×
of NumPy raw throughput; `JaxBackend` GPU ≤ `NumpyBackend` CPU at N ≥ 32
2-D. Add backend-parametric performance claims.

**C7 — Field storage: MeshFunction backed by Tensor.** Replace closure-backed
`LazyMeshFunction` hot paths with `Tensor`-backed storage. Verify correctness
against existing convergence claims.

**C8 — Explicit time integrators.** `TimeIntegrator` ABC; `RungeKutta2` and
`RungeKutta4`. Backend-agnostic; operates on `Tensor`-valued state. Lane B
derivation: truncation error O(hᵖ), p = 2, 4, confirmed symbolically.

**C9 — HDF5 checkpoint/restart.** Write/read `Tensor`-valued fields with
provenance sidecars (git hash, timestamp, parameter record). GPU-written
checkpoints readable on CPU-only machines.

**Open questions — Epoch 3 design points:**

1. **`@jax.jit` scope.** The natural granularity for JIT in this architecture
   is the full solve loop (assemble → iterate → residual), not individual
   `Backend` method calls. But tracing the loop requires that all branches on
   tensor shape/rank be static. Decision deferred to C5.

2. **`set_default_backend` vs. solver-level override.** The current design
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
| 0 | Theory / Geometry | **Mathematical foundations. ✓** Layer architecture and symbolic-reasoning import boundary; `foundation/`, `continuous/`, `discrete/`, `geometry/` type hierarchies; `CellComplex`, `Mesh`, `StructuredMesh`, `MeshFunction`, `RestrictionOperator`; process discipline M0–M2. |
| 1 | Geometry / Validation | **Observational grounding. ✓** `EuclideanManifold`, `CartesianChart`, `CartesianMesh`; first `validation/` notebook (Schwarzschild spacetime, GPS time dilation); settles `SymbolicFunction` interface and `Point` type (M3). |
| 2 | Discrete | **FVM Poisson solver. ✓** `PoissonEquation`; `DiffusiveFlux(2,4)`; `FVMDiscretization` + `NumericalFlux` family; oracle-free convergence framework; SPD analysis; `LinearSolver` ABC with `DenseJacobiSolver` and `DenseLUSolver`; end-to-end O(hᵖ) convergence sweep. FVM machinery reused from Epoch 5 onward. |
| 3 | Computation | **Backend-agnostic computation layer.** `Tensor` (arbitrary rank, `Real` protocol); `Backend` protocol with `PythonBackend`, `NumpyBackend`, `JaxBackend`; JIT-compiled solve loop; `TimeIntegrator` (RK2/RK4); field storage; HDF5 checkpoint/restart. In progress. |

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

---

## Platform milestones

| Milestone | Capability |
|-----------|------------|
| M0 | Process discipline: branch/PR/commit/attribution standards. ✓ |
| M1 | Verification infrastructure: convergence testing helpers, externally-grounded test pattern. ✓ |
| M2 | Documentation architecture: all live architectural decisions in `ARCHITECTURE.md`; `docs/` as API reference index. ✓ |
| M3 | Executable mathematical narrative: first `validation/` implementations (Schwarzschild spacetime, GPS time dilation); notebooks in `docs/` that import from `validation/` and run in CI. Settles coordinate-to-chart binding and the `SymbolicFunction` interface on concrete fields. ✓ |
| M4 | Validation infrastructure: manifests, provenance sidecars, comparison-result schema. Planned alongside Epoch 3. |
| M5 | Reproducibility capsule tooling: self-executing builder. |
| M6 | Application-repo capsule integration and multi-repository evidence regeneration. |
