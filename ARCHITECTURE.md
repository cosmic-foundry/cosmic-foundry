# Cosmic Foundry — Architecture

This document is the authoritative record of live architectural decisions
for this repository. Each decision is one paragraph. Decisions not yet
made are listed under *Open questions*. The code is the authoritative
description of any module's current design; this file records only what
is not self-evident from reading the code. `STATUS.md` is the navigation
anchor for the codebase. `DEVELOPMENT.md` covers workflow and process
decisions (including physics capability lanes).

---

## Technology baseline

**Python-only engine.** No compiled extensions are shipped from this
repository. Any native code the engine executes is produced at runtime
by a code-generation backend. `pybind11` and `ctypes` are emergency
escape hatches only; adopting either requires a documented justification
here. Pre-built libraries (JAX, NumPy, h5py) are consumed as
dependencies, not produced by this build.

**JAX + XLA as the primary kernel backend.** Every kernel shipped so far
is authored as a JAX kernel. The kernel interface (`Stencil`, `Reduction`
— see `computation/`) is structured so that secondary backends can be
added without changing call sites. Secondary backends (Numba, Taichi,
NVIDIA Warp, Triton) are listed as optional extras in `pyproject.toml`
but no adapter is written or exercised. Adding a secondary backend in
production requires documenting the workload and performance gap here.

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

**Visualization stack.** Field data is written in HDF5 (current `io/`)
and Zarr v3 (planned). Browser rendering uses WebGPU primary with a
WebGL2 fallback; desktop rendering uses pyvista/vispy for local
inspection. All colormaps are perceptual (cmasher, cmocean); rainbow/jet
are prohibited. Visual regression tests use pytest-mpl with SSIM
comparison.

---

## Mathematical hierarchy

The design principle governing `theory/` and `computation/`:

> Every class at the top level of `computation/` is exactly one step
> removed from `theory/` — it directly inherits from an ABC. Any class
> that is two steps removed must live one level down within `computation/`.

The `theory/` package defines pure mathematical ABCs with no JAX
dependency. The `computation/` package contains the first concrete
implementations. The current ABC hierarchy:

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

**Kernel abstraction (Op / Region / Policy / Dispatch).** The kernel
layer separates four independent axes: Op (the computation), Region (the
spatial domain), Policy (the execution strategy), and Dispatch (the
assembly and run). Any Op can be composed with any Region under any
Policy without changing call sites. `Stencil` (pointwise) and `Reduction`
(fold) are the two concrete Op types; `Extent` is the Region type;
`FlatPolicy` is the only implemented Policy. Dispatch runs the Op over
the Region via `op.execute(region, policy)`.

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

## Platform and application split

Cosmic Foundry is the **organizational platform**. Application
repositories — covering stellar physics, cosmology, galactic dynamics,
planetary formation, and other domains — build on top of it.

- Reusable computation infrastructure (kernels, mesh, fields, I/O,
  diagnostics) belongs here.
- Domain-specific physics implementations and observational validation
  data belong in application repos.
- Cross-scale workflows that compose two or more application domains
  belong in their own repository.

---

## Open architectural questions

These are decisions we know we need to make but have not yet made.
When a question is resolved, move it into the appropriate section above
and update the affected modules.

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
