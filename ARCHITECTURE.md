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

**The architecture is defined independently of any implementation
language.** The current Python implementation is one realization of the
architecture, not the architecture itself.

**The mathematical language of the architecture is differential geometry
on spatio-temporal manifolds, with PDE theory as the application layer.**
*(Tension: the current implementation uses only spatial manifolds;
no temporal or spacetime computations are implemented yet.)*

**Physical quantities are represented as instances of formal mathematical
abstractions.** Any concrete representation is an implementation detail.
*(Current inconsistency: `Field` and its subclasses are defined in
`theory/` but are not yet used in computation.)*

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

**The architecture is scale-agnostic.** The physics implementation is
independent of the deployment scale; any difference in outputs across
scales is attributable to non-deterministic order of operations, not to
architectural constraints.

**The engine is dimensionless internally.** Units are attached at the
boundary where results are compared against analytical solutions or
observational data.

---

## Technology baseline

**Python-only engine.** No compiled extensions are shipped from this
repository. Any native code the engine executes is produced at runtime
by a code-generation backend. `pybind11` and `ctypes` are emergency
escape hatches only; adopting either requires a documented justification
here. Pre-built libraries (NumPy, h5py) are consumed as
dependencies, not produced by this build.

**float64 as the default precision.** All field arrays default to
`float64`. Precision exceptions must be explicit and documented.

**Python ≥ 3.11.** Single source language end-to-end.

**Sphinx + MyST-NB documentation stack.** All narrative documentation is
built with Sphinx + MyST-NB. Docstrings follow the NumPy convention. The
docs build runs with warnings-as-errors; GitHub Actions deploys to GitHub
Pages. Sphinx-design provides layout components.

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

**`foundation/` is strictly third-party-free.** Primitive mathematical
abstractions shared by all layers: `Set`, `Function`, `IndexedSet`,
`IndexedFamily`. May only import from the Python standard library.

**`continuous/` is strictly third-party-free.** Manifolds, fields,
operators, boundary conditions. Imports only from `foundation/` and the
standard library. Enforced alongside `foundation/` by
`tests/test_theory_no_third_party_imports.py`.

**`discrete/` is strictly third-party-free.** Scheme description on finite
index sets. Imports from `foundation/` (vertical, is-a) and optionally from
`continuous/` (horizontal, has-a) via the `approximates` property. Enforced
alongside `foundation/` and `continuous/` by
`tests/test_theory_no_third_party_imports.py`. The `approximates`
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
├── SmoothManifold      — smooth (C∞) structure
│   └── PseudoRiemannianManifold — indefinite metric; free: signature, derived: ndim = sum(signature)
│       ├── RiemannianManifold   — positive-definite; free: ndim, derived: signature = (ndim, 0)
│       └── FlatManifold         — zero curvature
│           ├── EuclideanSpace   — ℝⁿ; free: ndim
│           └── MinkowskiSpace   — signature (1,3); no free parameters
└── ManifoldWithBoundary — has ∂M; interface: boundary → tuple[ManifoldWithBoundary, ...]
    └── Region           — compact, connected Ω ⊂ M; interface: ambient_manifold → SmoothManifold; derived: ndim

Field(Function)         — f: M → V on any Manifold; interface: manifold → Manifold
└── TensorField         — manifold narrows to SmoothManifold; interface: tensor_type → (p, q)
    ├── VectorField          — (1, 0); codomain TM; contravariant, not a form
    ├── SymmetricTensorField — (0, 2); g_{ij} = g_{ji}
    │   └── MetricTensor     — g on a PseudoRiemannianManifold; manifold narrows from SmoothManifold
    └── DifferentialForm     — (0, k); antisymmetric; interface: degree → k; tensor_type derived
        ├── ScalarField      — Ω⁰(M) = C∞(M); degree 0, tensor type (0, 0)
        └── CovectorField    — Ω¹(M) = Γ(T*M); degree 1, tensor type (0, 1)

DifferentialOperator(Function[Field, Field]) — L: Field → Field; interface: manifold → SmoothManifold, order → int

BoundaryCondition(Function)
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

**`BoundaryCondition` hierarchy.** Three ABCs in `theory/`:
`BoundaryCondition(Function)` is the blank root. `LocalBoundaryCondition`
represents `α·f + β·∂f/∂n = g` on a single face — abstract properties
`alpha: float`, `beta: float`, `constraint: Field`; covers Dirichlet
(`α=1, β=0`), Neumann (`α=0, β=1`), and Robin. `NonLocalBoundaryCondition`
is also blank beyond the root — it signals that the constraint depends on
field values outside the immediate neighborhood of the boundary point, but
makes no claim about the form of that non-locality;
concrete subclasses declare whatever geometric references they need. The
codimension-1 invariant is enforced structurally: every face of a
`ManifoldWithBoundary` has `ndim = parent.ndim - 1`.

**Derivation chain across the pseudo-Riemannian hierarchy.** At each
level, tighter constraints allow more to be derived:
- `SmoothManifold`: `ndim` is the free parameter (topologically primitive)
- `PseudoRiemannianManifold`: `signature` is the free parameter; `ndim = sum(signature)`
- `RiemannianManifold`: `ndim` is the free parameter; `signature = (ndim, 0)` enforces q = 0

**`intersect` on `IndexedSet`.** Set intersection is a fundamental
operation on any indexed set and lives as an `@abstractmethod` on
`IndexedSet`.

---


## Open architectural questions

These are decisions we know we need to make but have not yet made.
When a question is resolved, move it into the appropriate section above
and update the affected modules.

**Kernel composition model.**
A backend-agnostic interface separating kernel computation (Op) from
spatial domain and execution policy (Policy) is a design goal.
The earlier Op/Region/Policy/Dispatch framing was dropped before it was
realized. The current `Stencil` and `Reduction` primitives expose
`execute` directly; the formal model governing composition, backend
substitutability, and dispatch is unsettled.

**`DynamicManifold` for full GR.**
Full GR simulations cannot use a fixed-metric manifold: the metric
tensor `g_μν` is the dynamical variable evolved by the Einstein
equations. Planned: `DynamicManifold(PseudoRiemannianManifold)` in
`theory/` — signature is fixed (Lorentzian for GR), but the metric is
a field in the simulation state. In the 3+1 (ADM) formalism the
computational domain is a 3-D Riemannian spatial hypersurface; the
3-metric `γ_ij` and extrinsic curvature `K_ij` are evolved fields.
The concrete entry would be `Spacetime3Plus1(DynamicManifold)`.

**Halo fill fence.**
The halo-fill operation — ghost-cell exchange for stencil footprints
that cross patch boundaries — has a designed but not yet implemented
interface: `HaloFillFence` as a descriptor and `HaloFillPolicy` as the
execution unit. The driver inserts fences before dispatches whose stencil
radii exceed the local interior. The design is settled; implementation is
Epoch 2.

**Numerical transcription discipline.**
Physics capabilities sourced from reference tables (EOS polynomial fits,
reaction networks, opacity tables) need a discipline governing how
numeric tables are transcribed, verified, and updated independently of
the derivation-first lane policy. This decision is deferred to Epoch 7
(microphysics), when the first such capability lands.
