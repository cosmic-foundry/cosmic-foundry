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

**Every numerical method is verified against an analytical solution,
with the verification test living in this repository.**

**Where external data sources are ingested the uncertainty in that
data is explicitly quantified and propagated.**

**The engine is dimensionless internally.**

---

## Layer architecture

The codebase is organized into four packages with a strict dependency order:

```
foundation/   ←  continuous/
     ↑                ↑ (has-a, optional)
     └── discrete/ ───┘
              ↑
        computation/
```

**`foundation/`, `continuous/`, and `discrete/` are symbolic-reasoning layers.**
Their shared identity: they describe mathematical structure symbolically, without
numerical evaluation. Their import boundary reflects that identity — they may
only import from the Python standard library, `cosmic_foundry`, or packages on
the approved symbolic-reasoning list. The approved list is `{sympy}`. Additions
require justification against the symbolic-reasoning identity; numerical
computation packages (JAX, NumPy, SciPy) are excluded by definition. Enforced
by `tests/test_theory_no_third_party_imports.py`.

### foundation/  · Epoch 1 ✓

```
Set
├── IndexedFamily   — finite collection indexed by {0,…,n-1}; interface: __getitem__, __len__
└── IndexedSet      — finite rectangular subset of ℤⁿ; interface: ndim, shape, intersect

Function[D, C]      — callable mapping domain D → codomain C
```

### continuous/  · Epoch 1 ✓

```
Manifold(Set)           — topological space with a smooth atlas; interface: ndim, atlas → Atlas
└── PseudoRiemannianManifold — Manifold + metric; free: signature, derived: ndim = sum(signature)
    │   interface: metric → MetricTensor (abstract)
    ├── RiemannianManifold   — positive-definite metric; free: ndim, derived: signature = (ndim, 0)
    │   └── EuclideanSpace  — ℝⁿ; free: ndim; metric: EuclideanMetric; atlas: one global IdentityChart
    └── MinkowskiSpace       — signature (1,3); no free parameters; metric: MinkowskiMetric; atlas: one global IdentityChart

Chart(Function)         — diffeomorphism φ: U → V; U ⊂ M open, V ⊂ ℝⁿ open
                          interface: domain → Manifold, codomain → EuclideanSpace, inverse → Function
    IdentityChart       — φ(p) = p; standard chart for globally-chartable manifolds

Atlas(IndexedFamily)    — collection of charts covering M; constitutes the smooth structure of M
                          interface: manifold → Manifold, __getitem__ → Chart, __len__
    SingleChartAtlas    — one global chart covers all of M (EuclideanSpace, MinkowskiSpace)

Field(Function)         — f: M → V on any Manifold; interface: manifold → Manifold
└── TensorField         — interface: tensor_type → (p, q)
    ├── VectorField          — (1, 0); codomain TM; contravariant, not a form
    ├── SymmetricTensorField — (0, 2); g_{ij} = g_{ji}
    │   └── MetricTensor     — g on a PseudoRiemannianManifold
    │       ├── EuclideanMetric  — g_ij = δ_ij; __call__ returns sympy.eye(n)
    │       └── MinkowskiMetric  — g = diag(+1,−1,−1,−1); __call__ returns sympy.diag(1,-1,-1,-1)
    └── DifferentialForm     — (0, k); antisymmetric; interface: degree → k; tensor_type derived
        ├── ScalarField      — Ω⁰(M) = C∞(M); degree 0, tensor type (0, 0)
        └── CovectorField    — Ω¹(M) = Γ(T*M); degree 1, tensor type (0, 1)

DifferentialOperator(Function[Field, Field]) — L: Field → Field; interface: manifold → Manifold, order → int

Constraint(ABC)              — abstract; support: Manifold (the geometric locus where the constraint is enforced)
└── BoundaryCondition        — support is ∂M
    ├── LocalBoundaryCondition    — α·f + β·∂f/∂n = g on a single face; properties: alpha, beta, constraint
    └── NonLocalBoundaryCondition — constraint depends on values outside the immediate neighborhood
```

**`Constraint` / `BoundaryCondition` hierarchy.** `LocalBoundaryCondition`
covers Dirichlet (`α=1, β=0`), Neumann (`α=0, β=1`), and Robin via the
unified `α·f + β·∂f/∂n = g` form. `NonLocalBoundaryCondition` makes no
claim about the form of the non-locality; concrete subclasses declare
whatever geometric references they need.

**Derivation chain across the pseudo-Riemannian hierarchy.** At each
level, tighter constraints allow more to be derived:
- `Manifold`: `ndim` and `atlas` are the free parameters
- `PseudoRiemannianManifold`: `signature` is the free parameter; `ndim = sum(signature)`; `metric` is abstract — every concrete subclass must supply one
- `RiemannianManifold`: `ndim` is the free parameter; `signature = (ndim, 0)` enforces q = 0
- `EuclideanSpace`: `metric = EuclideanMetric` (g_ij = δ_ij) is the quantitative distinguisher from a generic `RiemannianManifold`
- `MinkowskiSpace`: `metric = MinkowskiMetric` (g = diag(+1,−1,−1,−1)) is the quantitative distinguisher from a generic `PseudoRiemannianManifold`

**Long-term direction: names as tags, not types.** The named subclass
hierarchy (`EuclideanSpace`, `MinkowskiSpace`) is an intermediate step.
The destination is parameterized instances: a manifold is fully specified
by its mathematical content (metric, signature), and names like "Euclidean
space" are informal labels attached to instances rather than class-level
distinctions. As the hierarchy matures, named subclasses are replaced by
concrete parameterized implementations of the abstract ABCs, and the ABCs
themselves become the only types. No new named subclasses should be added
without a plan to dissolve them.

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

**Open questions**

**What is the formal PDE object in the continuous layer?**
Conservation laws like ∂ρ/∂t + ∇·(ρv) = 0 are statements about continuous
fields. Before discretizing, we may want to express them as formal objects in
`continuous/`. The right interface is unclear and may only become clear once we
have a working discretization to invert from.

**What do SymPy-backed continuous objects look like?**
Constant fields are resolved: `EuclideanMetric.__call__` returns `sympy.eye(n)`
and `MinkowskiMetric.__call__` returns `sympy.diag(1,-1,-1,-1)` — both
independent of position. The open case is coordinate-dependent fields: a
concrete `ScalarField` backed by a SymPy expression `f(x, y) = sin(πx)sin(πy)`
where the coordinate symbols `x, y` are tied to a specific chart. The interface
for coordinate-dependent SymPy-backed fields (evaluatable analytical forms,
coordinate-to-chart binding) is not yet designed.

### discrete/  · Epochs 2–3

```
DiscreteField(Function[IndexedSet, V])
    approximates: Optional[Field]           — None if primary object, set if approximating continuous field
├── DiscreteScalarField
│   approximates: Optional[ScalarField]
└── DiscreteVectorField
    approximates: Optional[VectorField]
```

The `approximates` property declares that the discrete object is a finite
approximation of the named continuous object, enabling automatic convergence
checks at the `computation/` layer. When `None`, the discrete object is a
primary mathematical object with no continuous antecedent.

**Planned** (Epoch 2): Cartesian grid as a concrete `IndexedSet` with coordinate
geometry; cell and face structure. `DiscreteScalarField` and `DiscreteVectorField`
backed by the grid.

**Planned** (Epoch 3): Discrete differential operators: stencil coefficients
derived from continuous operators via SymPy; truncation error verified
algebraically; formal operator composition on the grid.

**Open question**

**Is scheme choice a first-class concept?**
A finite-difference discretization of ∇² is a precise mathematical act: choose
a grid, choose an approximation order, derive stencil coefficients. The
`approximates` property establishes the has-a link between a discrete object and
its continuous counterpart, but does not make scheme choice (e.g. "second-order
centered finite difference of the Laplacian") a first-class object. An open
question is whether a formal `Discretization` — a callable that maps a
`DifferentialOperator` + grid + order to a discrete stencil — belongs in
`discrete/`, or whether scheme choice remains implicit in how discrete objects
are constructed. The chart on the ambient manifold provides the coordinate map
that grounds the derivation; a first-class `Discretization` would reference it.

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
the derivation-first lane policy. This decision is deferred to Epoch 7
(microphysics), when the first such capability lands.

---

## Current work

**M2.5 design session: mathematical narrative documentation.**
What does the first notebook look like, and how does it hook into CI? Concrete
questions to settle: (1) which concept is the right entry point — `Set` and
`Function`, or the manifold hierarchy? (2) what makes a notebook "tested" — does
it run SymPy derivations that assert results, does it instantiate the ABCs, or
both? (3) how do we structure the `docs/` tree so notebooks accumulate without
becoming a maze? The goal is a format that can be repeated for every new concept
added to `continuous/` and `discrete/` as the epochs proceed.

**Epoch 2 design session: how do physical coordinates attach to a grid?**
The first concrete implementation is a Cartesian grid (`CartesianGrid` as a
concrete `IndexedSet` with coordinate geometry). The chart formalism is in
place: `Chart(Function)` maps manifold points to ℝⁿ, and `EuclideanSpace` carries
a `SingleChartAtlas`. But a `CartesianGrid` is a concrete `IndexedSet`, not a
manifold — so a `Chart` cannot directly act on it. The design question is: what
object maps grid indices to physical coordinates, and how does it relate to the
chart on the ambient `EuclideanSpace`? Settling this unblocks the `approximates`
convergence-check infrastructure (evaluate the continuous field at cell
coordinates, compare) and the SymPy-backed field interface (what coordinate
symbols does the expression use?).

---

## Physics roadmap

Each physics epoch adds new fields and equations to the continuous layer and
extends the discrete and numerical layers minimally to evaluate them.

### Foundation epochs

| Epoch | Layer | Capability |
|-------|-------|------------|
| 0 | — | Project scaffolding: CI, pre-commit, documentation standards. ✓ |
| 1 | Continuous | `continuous/` ABCs: full manifold and field hierarchy, operators, boundary conditions, metric; coordinate structure (`Chart`, `Atlas`, `IdentityChart`, `SingleChartAtlas`); `SmoothManifold.atlas` constitutive. `foundation/` ABCs: `Set`, `Function`, `IndexedSet`, `IndexedFamily`. `discrete/` ABCs: `DiscreteField`, `DiscreteScalarField`, `DiscreteVectorField`. ✓ |
| 2 | Discrete | Cartesian grid as a concrete `IndexedSet` with coordinate geometry; cell and face structure. `DiscreteScalarField` and `DiscreteVectorField` backed by the grid. |
| 3 | Discrete | Discrete differential operators: stencil coefficients derived from continuous operators via SymPy; truncation error verified algebraically; formal operator composition on the grid. |
| 4 | Numerical | JAX evaluation layer: concrete field storage as `jax.Array`; JIT-compiled stencil application; explicit time integration; HDF5 I/O with provenance. |

### Physics epochs

| Epoch | Capability |
|-------|------------|
| 5 | Scalar transport: linear advection and diffusion on a Cartesian grid. First end-to-end simulation; validates the full pipeline. |
| 6 | Newtonian hydrodynamics: Euler equations, finite-volume Godunov, PPM reconstruction, HLLC/HLLE Riemann solvers. |
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

---

## Platform milestones

| Milestone | Capability |
|-----------|------------|
| M0 | Process discipline: branch/PR/commit/attribution standards. ✓ |
| M1 | Verification infrastructure: convergence testing helpers, externally-grounded test pattern. ✓ |
| M2 | Documentation architecture: all live architectural decisions in `ARCHITECTURE.md`; `docs/` as API reference index. ✓ |
| M2.5 | Mathematical narrative documentation: executable MyST-NB notebooks that explain each layer of the hierarchy — what the formal concepts mean, how they relate, and why the code is structured the way it is. Notebooks run in CI, so every mathematical claim is machine-checked. |
| M3 | Validation infrastructure: manifests, provenance sidecars, comparison-result schema. Planned alongside Epoch 4. |
| M4 | Reproducibility capsule tooling: self-executing builder. |
| M5 | Application-repo capsule integration and multi-repository evidence regeneration. |

### Per-epoch verification standard

Every physics epoch must satisfy this checklist before it is considered verified:

- Derivation document with SymPy checks for any new numerical scheme (Lanes B and C)
- At least one externally-grounded convergence test against an analytical solution
  or observational data (not an engine-generated golden file); where an analytical
  solution exists, the relevant `DiscreteField.approximates` is set so the check
  runs automatically
- Lane A/B/C classification stated in the PR description
