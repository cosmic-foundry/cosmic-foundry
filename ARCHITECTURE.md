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
`theory/` but are not used in computation. Kernel inputs and outputs are
raw JAX arrays wrapped in `Array[T]`, not `Field` instances.)*

**Every numerical method is formally derived from its continuous
mathematical counterpart.** The derivation is machine-checkable (SymPy)
except where the argument is geometric or topological, in which case a
human-readable derivation is required. Derivations live in `derivations/`.
*(Current inconsistency: the `derivations/` directory does not exist.
The Laplacian stencil has no formal derivation document.)*

**Every numerical method is verified against an analytical solution or
observational data, with the verification test living in this
repository.**
*(Current inconsistency: the convergence testing infrastructure exists
in `tests/utils/convergence.py` but is not applied to any production
operator. The Laplacian is checked only on a polynomial for which the
stencil is exact — not a convergence test.)*

**Where external data sources are ingested** (reaction rates, opacity
tables, observational measurements), **the uncertainty in that data is
explicitly quantified and propagated.**

**The architecture is scale-agnostic.** The physics implementation is
independent of the deployment scale; any difference in outputs across
scales is attributable to non-deterministic order of operations, not to
architectural constraints.

**Every file the engine writes carries provenance metadata** identifying
the exact repository state from which it was generated.
*(Current inconsistency: `io/` writes HDF5 files with no git commit
hash or repository state attached. The `Provenance` class in
`manifests/` covers external data ingestion only.)*

**The engine is dimensionless internally.** Units are attached at the
boundary where results are compared against analytical solutions or
observational data.

---

## Technology baseline

**Python-only engine.** No compiled extensions are shipped from this
repository. Any native code the engine executes is produced at runtime
by a code-generation backend. `pybind11` and `ctypes` are emergency
escape hatches only; adopting either requires a documented justification
here. Pre-built libraries (JAX, NumPy, h5py) are consumed as
dependencies, not produced by this build.

**JAX + XLA is the current kernel backend.** Every kernel is authored
as a JAX kernel. The formal model governing backend substitutability and
kernel composition is an open question (see *Open architectural
questions*).

**`jax.distributed` + NCCL/GLOO for host parallelism.** No MPI layer in
the baseline. `mpi4py` is available as an optional extra for sites where
`jax.distributed` cannot initialize over the native interconnect.

**float64 as the default precision.** All field arrays default to
`float64`. Precision exceptions must be explicit and documented.

**Python ≥ 3.11.** Single source language end-to-end.

**Sphinx + MyST-NB documentation stack.** All narrative documentation is
built with Sphinx + MyST-NB. Docstrings follow the NumPy convention. The
docs build runs with warnings-as-errors; GitHub Actions deploys to GitHub
Pages. Sphinx-design provides layout components.

---

## Mathematical hierarchy

**`theory/` is strictly third-party-free.** The `theory/` package
defines pure mathematical ABCs and may not import from any package
outside the Python standard library. Mathematical concreteness (classes
parameterized by Python primitives) belongs in `theory/`; computational
concreteness (JAX, NumPy, HDF5) belongs outside it. Enforced by
`tests/test_theory_no_third_party_imports.py`.

**`computation/` contains distance-1 implementations.** Every class at
the top level of `computation/` directly inherits from an ABC in
`theory/`. Classes two or more steps removed live one level down within
their package.

The current ABC hierarchy:

```
Set
├── IndexedFamily           — finite collection indexed by {0,…,n-1}; interface: __getitem__, __len__
│   └── Array[T]            (computation/) — tuple-backed finite indexed family
├── IndexedSet              — finite rectangular subset of ℤⁿ; interface: ndim, shape, intersect
│   └── Extent              (computation/) — half-open integer index extent
│   └── Discretization      — IndexedSet approximating functions on a manifold
│       └── LocatedDiscretization — DOFs at specific points; interface: node_positions
│           └── Patch       (mesh/) — uniform Cartesian LocatedDiscretization
└── Manifold                — topological manifold; interface: ndim
    ├── SmoothManifold      — smooth (C∞) structure
    │   └── PseudoRiemannianManifold — indefinite metric; free: signature, derived: ndim = sum(signature)
    │       ├── RiemannianManifold   — positive-definite; free: ndim, derived: signature = (ndim, 0)
    │       └── FlatManifold         — zero curvature
    │           ├── EuclideanSpace   (theory/) — ℝⁿ; free: ndim
    │           └── MinkowskiSpace   (theory/) — signature (1,3); no free parameters
    └── ManifoldWithBoundary — has ∂M; interface: boundary → tuple[ManifoldWithBoundary, ...]
        └── Domain           (geometry/) — finite region of a SmoothManifold with origin and size

Function                — f: A × Θ → B; interface: execute
├── Stencil             (computation/) — parametric pointwise stencil; parameters: fn, radii
├── Reduction           (computation/) — parametric field fold; parameters: operator, identity
└── PartitionDomain     (mesh/) — partitions a Domain into an Array[Patch]
```

**`BoundaryCondition` hierarchy.** Three ABCs in `theory/`:
`BoundaryCondition(Function)` is the blank root. `LocalBoundaryCondition`
represents `α·f + β·∂f/∂n = g` on a single face — abstract properties
`alpha: float`, `beta: float`, `constraint: Field`; covers Dirichlet
(`α=1, β=0`), Neumann (`α=0, β=1`), and Robin. `NonLocalBoundaryCondition`
is also blank beyond the root — it signals that the constraint depends on
field values outside the immediate neighborhood of the boundary point, but
makes no claim about the form of that non-locality;
concrete subclasses declare whatever geometric references they need
(`FaceIdentification` carries a pair of boundary faces; a Dirichlet-to-Neumann
map carries the full boundary). The codimension-1 invariant is enforced
structurally: every face in `Domain.boundary` has `ndim = parent.ndim - 1`.
Concrete subclasses with JAX-backed `execute` live in `computation/`.

**Derivation chain across the pseudo-Riemannian hierarchy.** At each
level, tighter constraints allow more to be derived:
- `SmoothManifold`: `ndim` is the free parameter (topologically primitive)
- `PseudoRiemannianManifold`: `signature` is the free parameter; `ndim = sum(signature)`
- `RiemannianManifold`: `ndim` is the free parameter; `signature = (ndim, 0)` enforces q = 0

**`intersect` on `IndexedSet`.** Set intersection is a fundamental
operation on any indexed set and lives as an `@abstractmethod` on
`IndexedSet`. `Extent` implements it directly; `Patch` delegates to
`self.index_extent.intersect(other)`.

---

## Operator model

**Kernel primitives.** `Stencil` (pointwise) and `Reduction` (fold) are
the two concrete kernel types, both in `computation/`. Each exposes an
`execute` method. A formal model separating the kernel computation from
the spatial region and execution policy is a design goal but is not yet
realized; see *Open architectural questions*.

**Global reduction primitive.** `Reduction(operator, identity)` is the
primitive for field-level folds. It returns a 0-dimensional JAX array
rather than a Python scalar so XLA can fuse the reduction into
surrounding computation. The identity element is required for correctness
under `jit`.

**Operator documentation convention.** Every operator class carries a
structured docstring block declaring its mathematical contract.
`Function:` blocks state domain, codomain, and approximation parameters
Θ and order p. `Source:` blocks document reads from external state
(files, network). `Sink:` blocks document writes to external state. This
convention makes each class's contract auditable without reading its
implementation.

---


## Open architectural questions

These are decisions we know we need to make but have not yet made.
When a question is resolved, move it into the appropriate section above
and update the affected modules.

**Kernel composition model.**
A backend-agnostic interface separating kernel computation (Op) from
spatial region (Region) and execution policy (Policy) is a design goal.
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
The concrete geometry entry is `Spacetime3Plus1(DynamicManifold)` in
`geometry/`.

**Domain as Array[Domain].**
`Domain` is currently a single bounded region of a manifold. Multi-patch
or non-rectangular simulation domains may eventually require `Domain` to
be an `Array[Domain]` — a finite indexed family of sub-domains — rather
than a single object. If so, `PartitionDomain` dissolves: the domain
decomposition IS the domain. This generalization is deferred until a
concrete use case requires it; the single-`Domain` design is coherent for
all planned physics capabilities.

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
