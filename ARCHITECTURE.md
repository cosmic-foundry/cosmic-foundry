# Cosmic Foundry — Architecture

This document is the authoritative record of live architectural decisions
for this repository. Decisions not yet made are listed under *Open
questions*. The code is designed to make its own structure self-evident;
ARCHITECTURE.md records the decisions and reasoning behind that
structure, not a description of it. `DEVELOPMENT.md` covers workflow and
process decisions (including physics capability lanes).

---

## Architectural basis

These are the foundational claims about this repository. Each is a
commitment: the code must satisfy it, tests enforce it where possible,
and any PR that violates a claim must explicitly revise it here rather
than quietly breaking it.

**Cosmic Foundry is a general-purpose PDE simulation engine, optimized
for astrophysical use cases.** It provides reusable computation
infrastructure — kernels, mesh, fields, I/O, diagnostics, manifest
tooling — on which application repositories build domain-specific
physics. No domain-specific physics implementation and no observational
validation data belongs here.

**The mathematical language of the architecture is differential geometry
on spatio-temporal manifolds, with PDE theory as the application layer.**

**Physical quantities are represented as instances of formal mathematical
abstractions.** Any concrete representation is an implementation detail.

**Every numerical method is formally derived from its continuous
mathematical counterpart.** The derivation is machine-checkable (SymPy)
except where the argument is geometric or topological, in which case a
human-readable derivation is required. Derivations are documented in the
modules that implement the methods; SymPy is never imported at module load time.

**Every numerical method is verified against an analytical solution or
observational data, with the verification test living in this
repository.**

**Where external data sources are ingested** (reaction rates, opacity
tables, observational measurements), **the uncertainty in that data is
explicitly quantified and propagated.**

**The engine is dimensionless internally.** Units are attached at the
boundary where results are compared against analytical solutions or
observational data.

---

## Technology baseline

**Python-only engine.** No compiled extensions are shipped from this
repository. Any native code the engine executes is produced at runtime
by a code-generation backend. `pybind11` and `ctypes` are emergency
escape hatches only; adopting either requires a documented justification
here. Pre-built numerical libraries are consumed as
dependencies, not produced by this build.

**float64 as the default precision.** All field arrays default to
`float64`. Precision exceptions must be explicit and documented.

**Python ≥ 3.11.** Single source language end-to-end.

**Sphinx + MyST-NB documentation stack.** All narrative documentation is
built with Sphinx + MyST-NB. Docstrings follow the NumPy convention. The
docs build runs with warnings-as-errors. Sphinx-design provides layout components.

---

## Package structure and boundaries

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

- **`foundation/`** — `Set`, `Function`, `IndexedSet`, `IndexedFamily`.
- **`continuous/`** — manifolds, fields, operators, boundary conditions.
- **`discrete/`** — scheme description; imports `foundation/` (is-a) and
  optionally `continuous/` (has-a) via `approximates`. The `approximates`
property on each discrete type is `Optional[<continuous counterpart>]`:
when set, it declares that the discrete object is a finite approximation
of the named continuous object, enabling automatic convergence checks at
the `computation/` layer. When `None`, the discrete object is a primary
mathematical object with no continuous antecedent.

**`computation/`** — JAX evaluation. The only layer that touches floats.

## Mathematical hierarchy

**`foundation/` types:**

```
Set
├── IndexedFamily   — finite collection indexed by {0,…,n-1}; interface: __getitem__, __len__
└── IndexedSet      — finite rectangular subset of ℤⁿ; interface: ndim, shape, intersect

Function[D, C]      — callable mapping domain D → codomain C
```

**`continuous/` types:**

```
Manifold(Set)
├── SmoothManifold      — smooth (C∞) structure; atlas constitutes the smooth structure
│   │   interface: ndim, atlas → Atlas
│   └── PseudoRiemannianManifold — indefinite metric; free: signature, derived: ndim = sum(signature)
│       ├── RiemannianManifold   — positive-definite; free: ndim, derived: signature = (ndim, 0)
│       │   └── EuclideanSpace  — ℝⁿ; free: ndim; atlas: one global IdentityChart
│       └── MinkowskiSpace       — signature (1,3); no free parameters; atlas: one global IdentityChart

Chart(Function)         — diffeomorphism φ: U → V; U ⊂ M open, V ⊂ ℝⁿ open
                          interface: domain → SmoothManifold, codomain → EuclideanSpace, inverse → Function
    IdentityChart       — φ(p) = p; standard chart for globally-chartable manifolds

Atlas(IndexedFamily)    — collection of charts covering M; constitutes the smooth structure of M
                          interface: manifold → SmoothManifold, __getitem__ → Chart, __len__
    SingleChartAtlas    — one global chart covers all of M (EuclideanSpace, MinkowskiSpace)

Field(Function)         — f: M → V on any Manifold; interface: manifold → Manifold
└── TensorField         — manifold narrows to SmoothManifold; interface: tensor_type → (p, q)
    ├── VectorField          — (1, 0); codomain TM; contravariant, not a form
    ├── SymmetricTensorField — (0, 2); g_{ij} = g_{ji}
    │   └── MetricTensor     — g on a PseudoRiemannianManifold; manifold narrows from SmoothManifold
    └── DifferentialForm     — (0, k); antisymmetric; interface: degree → k; tensor_type derived
        ├── ScalarField      — Ω⁰(M) = C∞(M); degree 0, tensor type (0, 0)
        └── CovectorField    — Ω¹(M) = Γ(T*M); degree 1, tensor type (0, 1)

DifferentialOperator(Function[Field, Field]) — L: Field → Field; interface: manifold → SmoothManifold, order → int

Constraint(ABC)              — abstract; support: Manifold (the geometric locus where the constraint is enforced)
└── BoundaryCondition        — support is ∂M
    ├── LocalBoundaryCondition    — α·f + β·∂f/∂n = g on a single face; properties: alpha, beta, constraint
    └── NonLocalBoundaryCondition — constraint depends on values outside the immediate neighborhood
```

**`discrete/` types:**

```
DiscreteField(Function[IndexedSet, V])
    approximates: Optional[Field]           — None if primary object, set if approximating continuous field
├── DiscreteScalarField
│   approximates: Optional[ScalarField]
└── DiscreteVectorField
    approximates: Optional[VectorField]
```

**`Constraint` / `BoundaryCondition` hierarchy.** `LocalBoundaryCondition`
covers Dirichlet (`α=1, β=0`), Neumann (`α=0, β=1`), and Robin via the
unified `α·f + β·∂f/∂n = g` form. `NonLocalBoundaryCondition` makes no
claim about the form of the non-locality; concrete subclasses declare
whatever geometric references they need.

**Derivation chain across the pseudo-Riemannian hierarchy.** At each
level, tighter constraints allow more to be derived:
- `SmoothManifold`: `ndim` is the free parameter (topologically primitive)
- `PseudoRiemannianManifold`: `signature` is the free parameter; `ndim = sum(signature)`
- `RiemannianManifold`: `ndim` is the free parameter; `signature = (ndim, 0)` enforces q = 0

---


## Open architectural questions

These are decisions we know we need to make but have not yet made.
When a question is resolved, move it into the appropriate section above
and update the affected modules.

**Kernel composition model.**
A backend-agnostic interface separating kernel computation (Op) from
spatial domain and execution policy (Policy) is a design goal. An
earlier Op/Policy/Dispatch framing was dropped before it was realized.
The formal model governing composition, backend substitutability, and
dispatch is unsettled.

**`DynamicManifold` for full GR.**
Full GR simulations cannot use a fixed-metric manifold: the metric
tensor `g_μν` is the dynamical variable evolved by the Einstein
equations. Planned: `DynamicManifold(PseudoRiemannianManifold)` in
`continuous/` — signature is fixed (Lorentzian for GR), but the metric is
a field in the simulation state. In the 3+1 (ADM) formalism the
computational domain is a 3-D Riemannian spatial hypersurface; the
3-metric `γ_ij` and extrinsic curvature `K_ij` are evolved fields.
The concrete entry would be `Spacetime3Plus1(DynamicManifold)`.

**Numerical transcription discipline.**
Physics capabilities sourced from reference tables (EOS polynomial fits,
reaction networks, opacity tables) need a discipline governing how
numeric tables are transcribed, verified, and updated independently of
the derivation-first lane policy. This decision is deferred to Epoch 7
(microphysics), when the first such capability lands.

**Is scheme choice a first-class concept?**
A finite-difference discretization of ∇² is a precise mathematical act: choose
a grid, choose an approximation order, derive stencil coefficients. The
`approximates` property on `DiscreteField` establishes the has-a link between
a discrete object and its continuous counterpart, but does not make scheme choice
(e.g. "second-order centered finite difference of the Laplacian") a first-class
object. An open question is whether a formal `Discretization` — a callable that
maps a `DifferentialOperator` + grid + order to a discrete stencil — belongs in
`discrete/`, or whether scheme choice remains implicit in how discrete objects
are constructed. The chart on the ambient manifold provides the coordinate map
that grounds the derivation; a first-class `Discretization` would reference it.

**What is the formal PDE object in the continuous layer?**
Conservation laws like ∂ρ/∂t + ∇·(ρv) = 0 are statements about continuous
fields. Before discretizing, we may want to express them as formal objects in
`continuous/`. The right interface is unclear and may only become clear once we
have a working discretization to invert from.

**What do SymPy-backed continuous objects look like?**
The symbolic-reasoning identity makes SymPy available in `continuous/` and
`discrete/`. The natural use is analytical field representations — a concrete
`ScalarField` backed by a SymPy expression `f(x, y) = sin(πx)sin(πy)` — which
would make `approximates` algebraically live: stencil derivation and truncation
error analysis could be done in code rather than in documentation. The coordinate
symbols `x, y` in such an expression are tied to a specific chart: the chart's
component functions x¹, …, xⁿ are exactly the coordinate symbols the expression
uses. The interface for SymPy-backed fields (evaluatable analytical forms,
coordinate-to-chart binding) is not yet designed.
