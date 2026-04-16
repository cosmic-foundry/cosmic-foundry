# ADR-0010 — Kernel abstraction model: Op / Region / Policy / Dispatch

- **Status:** Proposed
- **Date:** 2026-04-15

## Context

ADR-0002 committed to a `@kernel` descriptor layer that would allow
secondary backends to register adapters without changing call sites.
That decision left the internal structure of the descriptor unspecified,
deferring it to Epoch 1 design.

Three subsequent design observations sharpened what the descriptor must
express:

**1. Per-element operations are never called in isolation.** In a real
multiphysics code, EOS evaluations, Riemann solvers, and reconstruction
operators are always called within a sweep over a mesh or particle set.
This means the unit of dispatch is not the per-element function but the
iteration over a domain that inlines one or more per-element functions.
The two must be separate abstractions.

**2. Kernel-launch granularity must be independently tunable.** Whether
reconstruction and the Riemann solve happen in one kernel launch or two
is a performance question whose answer varies by GPU generation, problem
size, and memory hierarchy. If the per-element computation and the
kernel boundary are expressed in the same abstraction, fusion
experiments require touching physics code. The correct design keeps
kernel boundary (a scheduling concern) separate from element-level
computation (a physics concern).

**3. Three axes are required, not two.** The research survey
(`research/01-frameworks.md` §1.2 Kokkos, §1.6 Singe) established that
performance-portable kernel programming requires three independently
variable concerns:

- **Computational** — what is computed per element.
- **Spatial** — what region of elements is iterated over, and how those
  elements are batched for GPU efficiency.
- **Execution** — how threads are organized to process the region:
  flat per-element, tiled with cooperative scratchpad, or
  warp-specialized for irregular workloads.

These axes must be independently specifiable because each has
independent tuning decisions. Prior designs conflated the spatial and
execution axes (e.g., a "Tile" concept that bundled both the spatial
sub-region and the cooperative thread group), or conflated the
computational and execution axes (e.g., a "kernel" that encoded both
the per-element function and its dispatch configuration).

## Decision

The kernel abstraction layer is structured around four top-level
concepts, each owning exactly one axis or one composition. Op carries a
subordinate `AccessPattern` descriptor because locality metadata is a
property of the per-element computation, not an independent dispatch
axis:

### AccessPattern (Op metadata)

Every Op declares an `access_pattern` attribute describing the Op's
locality contract. This is metadata on the computational axis: the Op
author declares it, while Region, Policy, and Dispatch consume it.
Epoch 1 defines one concrete subtype, `Stencil`, because the Epoch 1
exit criterion is a 3-D 7-point Laplacian. Non-stencil patterns remain
anticipated extensions, but their metadata is deliberately not frozen by
this ADR.

```python
class AccessPattern(ABC):
    @abstractmethod
    def halo_width(self, axis: int) -> int:
        """Ghost-cell depth the driver must fill on this axis."""
        ...

class Stencil(AccessPattern):
    """Fixed-width neighborhood centered on a grid element."""
    @classmethod
    def seven_point(cls) -> "Stencil": ...
    @classmethod
    def symmetric(cls, order: int) -> "Stencil": ...

    def halo_width(self, axis: int) -> int: ...   # returns stencil radius
```

The driver calls `op.access_pattern.halo_width(axis)` uniformly when
sizing halos; no Op-type special-casing is needed for stencil kernels.

Two non-stencil families are expected later, but they need more
metadata than the current Epoch 1 stencil contract:

- **Gather/scatter** needs separate read and write footprints, target
  field information, ownership rules, and write-conflict semantics
  (deposit vs. gather, combine operation, determinism requirements).
- **Reductions** need algebraic and visibility metadata: identity,
  combiner, result name or location, result shape and dtype, reduction
  scope, finalization behavior, and deterministic/distributed reduction
  expectations.

Those extensions should be added when particle or diagnostic workloads
force their concrete API shape. Until then, `access_pattern` means the
stencil locality contract required by Epoch 1.

### Op (computational axis)

An Op is any per-element callable satisfying the Op protocol: it
declares access metadata and implements a traceable `__call__`. It
describes *what* to compute at a single element given its local
neighborhood. It is not dispatchable on its own.

`Dispatch` accepts the structural contract, not only instances of one
nominal base class. The reference API provides two authoring forms:

- an `@op(...)` decorator for simple function-shaped Ops; and
- an optional `Op` abstract base class for class-based or
  parameterized Ops.

The protocol shape is:

```python
class OpLike(Protocol):
    access_pattern: AccessPattern

    reads:    tuple[str, ...] = ()
    writes:   tuple[str, ...] = ()
    backends: frozenset[Backend] = frozenset({Backend.JAX})

    def __call__(self, ...): ...
```

A simple function-shaped Op uses the decorator:

```python
@op(
    access_pattern=Stencil.seven_point(),
    reads=("phi",),
    writes=("laplacian_phi",),
)
def seven_point_laplacian(q, i, j, k): ...
```

A class-based Op uses the optional `Op` ABC when constructor state,
stable identity, or helper methods matter:

```python
class Op(ABC):
    @property
    @abstractmethod
    def access_pattern(self) -> AccessPattern: ...

    reads:    tuple[str, ...] = ()
    writes:   tuple[str, ...] = ()
    backends: frozenset[Backend] = frozenset({Backend.JAX})

    @abstractmethod
    def __call__(self, ...): ...
```

A parameterized Op declares `access_pattern` as a `@property` and
resolves it from constructor arguments:

```python
class Reconstruct(Op):
    def __init__(self, order: int):
        self.order = order

    @property
    def access_pattern(self) -> AccessPattern:
        return Stencil.symmetric(self.order)

    reads  = ("primitive_vars",)
    writes = ("edge_states",)

    def __call__(self, q, i, j, k): ...
```

A diagnostic or particle Op will use the same Op protocol once its
non-stencil access metadata is specified, but those concrete
`AccessPattern` subtypes are outside the Epoch 1 contract:

```python
# Future extension, not Epoch 1 API:
@op(access_pattern=ParticleDeposit(...), reads=("charge",))
def deposit_charge(q, i, j, k): ...

# Future extension, not Epoch 1 API:
@op(access_pattern=DiagnosticReduction(...), reads=("velocity",))
def compute_cfl(q, i, j, k) -> float: ...
```

Instantiation is specialization: `ppm = Reconstruct(order=2)`. Each
instance is a fully-specified Op with concrete metadata. This is the
Python/JAX analog of a C++ function template instantiation: `order`
is resolved before JAX traces `__call__`, producing an
order-specialized XLA computation per instance. Function-shaped Ops
are preferred for stateless or lightly parameterized computations;
class-based Ops are preferred when metadata depends on constructor
state or when the operation has a meaningful persistent identity.

**Metadata catalog for Epoch 1:**

| Attribute | Type | Required | Purpose |
|---|---|---|---|
| `access_pattern` | `AccessPattern` | Yes | Stencil halo width for Epoch 1; future extensions add particle and reduction metadata |
| `reads` | `tuple[str, ...]` | Epoch 2 | Named fields read; task graph derives data dependencies |
| `writes` | `tuple[str, ...]` | Epoch 2 | Named fields written; task graph derives data dependencies |
| `backends` | `frozenset[Backend]` | No | Execution contexts that can lower this Op |

Additional metadata (register pressure estimate, analytic solution for
test generation) may be added as the interface matures.

`reads` and `writes` are Epoch 1 authoring shorthand, not the final
granularity of the task graph. Before ordering work, the driver
normalizes Op metadata together with the Dispatch's Region and
AccessPattern into dependencies over a field, an iteration extent, an
access mode, and the expanded access footprint. In Epoch 1, stencil
patterns expand read footprints by their halo width. Future
gather/scatter and reduction patterns must represent read and write
footprints, conflict/combiner semantics, and externally visible
reduction results explicitly enough for the driver to order work
correctly. This avoids both false ordering from field-name-only
dependencies (for example, independent meshblocks of the same field)
and missing ordering from halo, neighbor, write-conflict, or
reduction-result dependencies.

Dispatch composition hides internal temporaries from the task graph.
Only external reads, externally visible writes, and externally consumed
reduction results become graph dependencies. In-place-looking updates
such as `reads=("U",), writes=("U",)` are interpreted by the driver as
versioned field transitions (for example, `U^n -> U^(n+1)`), not as
ambiguous mutation of one timeless field.

**`__call__` signature — decided.** An Op receives element indices
plus an array-like field argument. The field argument is *not*
required to be a raw JAX array; it is required to support
`__getitem__` with Region-coordinate semantics. A Field consists of
one or more FieldSegments, each pairing a payload with the Extent over
which that payload is valid; Placement maps SegmentIds to
process/device owners. The Op does not inspect FieldSegment or
Placement metadata directly. Dispatch validation checks that the
Field's placed segments cover the Region extent and access footprint
required by the Op. Under `FlatPolicy` the argument can be a local
FieldSegment payload or a thin Field view over one or more local
segments. Under `TiledPolicy` the Policy passes a `FieldView` wrapper
that intercepts `__getitem__` calls and redirects them to an
SRAM-backed halo-extended tile, translating Region coordinates to
tile-local offsets transparently. The Op's `__call__` code is
identical under both policies; the physics author is never aware of
segment boundaries, tile boundaries, or process ownership.

`FieldView` is a JAX pytree (registered via
`jax.tree_util.register_pytree_node`); JAX traces through its
`__getitem__` as ordinary index arithmetic on the underlying tile
array. `FieldView` is a `TiledPolicy` implementation concern and
is not part of the Epoch 1 deliverable.

- The optional `Op` abstract base class enforces the interface at
  import time for class-based Ops: a subclass that omits
  `access_pattern` or `__call__` raises `TypeError` on class
  construction, not at driver assembly. Function-shaped Ops are
  validated by the `@op(...)` decorator and by `Dispatch` construction.
- An Op's `__call__` must be traceable in the backend's execution
  context (JAX-jittable for the primary backend). Constructor
  parameters like `order` are resolved before tracing and do not
  appear in the traced computation.
- Examples: Helmholtz EOS evaluation, HLLC Riemann solver, PPM
  reconstruction, 7-point Laplacian stencil, cooling rate per cell,
  CFL computation, divergence norm check.

### Region (spatial / iteration axis)

A Region is the set of elements over which a Dispatch iterates. It may
be geometric (mesh cells, faces, edges, nodes) or non-geometric
(particles, neighbor-selected particle subsets). A Region owns two
iteration concerns:

- **Extent**: which elements are iterated over (a meshblock, a face
  array, a particle array, or another indexable collection).
- **Batching**: whether the Region is a single block or a packed
  collection of blocks (analogous to Parthenon's `MeshBlockPack`).
  Region batching determines how many elements enter a single kernel
  launch; the Op is unaware of the batch dimension.

In JAX, a batched Region maps to `vmap` over the Op; the Op
implementation is unchanged.

### Policy (execution axis)

How threads are organized to process a Region. A Policy is a
swappable object attached to a Dispatch. Three policies are anticipated:

| Policy | Description | When needed |
|---|---|---|
| `FlatPolicy` | One thread per element; same code for all. Kokkos `RangePolicy` analog. | Default; sufficient for Epochs 1–3. |
| `TiledPolicy` | Thread team with cooperative scratchpad load and explicit barrier. Kokkos `TeamPolicy` + `team_scratch` analog. Addresses bandwidth-bound stencils. | Epoch 2–3 when mesh stencil ops become real workloads. |
| `WarpSpecializedPolicy` | Warps within a thread block assigned to different code paths. Singe analog. Addresses compute-irregular workloads (stiff ODE, tree walks). | Epoch 6 when nuclear reaction networks demand it. |

Swapping a Policy requires no changes to the Op or to the Region. The
physics author does not choose a Policy; the driver or a performance
annotation does.

Reductions do not introduce a fourth public Policy. Performance
portability systems such as Kokkos, RAJA, SYCL, and OpenMP make
reductions explicit because the backend needs an identity, combiner,
result location, and sometimes finalization or location-tracking
semantics. Cosmic Foundry should follow that precedent when diagnostic
reductions become an implementation target: reduction metadata belongs
with the Op's access/output contract, while block-local, tree, warp, or
multi-stage reduction algorithms remain backend lowering strategies
inside `FlatPolicy`, `TiledPolicy`, or a future Policy, not
author-facing execution organizations.

### Dispatch (dispatch unit)

A Dispatch is the composition of one or more Ops, a Region, and a
Policy. One Dispatch equals one kernel launch — one `jit` boundary in
JAX. The Dispatch is the unit that the driver schedules, the task graph
depends on, and the profiler attributes. When multiple Ops are fused
into one Dispatch, Dispatch assembly derives the combined
AccessPattern and field dependencies from the contained Ops; the Ops
themselves remain unaware of fusion.

The physics author writes Ops. The driver assembles Dispatches. Fusion
experiments are expressed by composing or splitting Ops within a
Dispatch body, not by modifying Ops.

Separate Dispatches are not an author-facing explicit-fusion unit. A
future driver or backend may fuse adjacent compatible Dispatches as a
transparent optimization, provided observable semantics are unchanged.
Ordinary data dependencies order Dispatches but do not, by themselves,
force intermediate fields to materialize or prohibit such transparent
fusion.

When code needs a boundary that optimization must not cross, that
boundary is expressed as a driver/task-graph fence, not as an Op,
Region, or Policy feature. Fences are required for communication or
halo exchange, AMR synchronization, host-visible diagnostics or I/O,
externally consumed reduction results, profiling/timing boundaries, and
any case where an intermediate field must be materialized before later
work. This keeps Dispatch as the local lowering unit while the driver
owns global ordering, visibility, and synchronization semantics.

## Naming rationale and alternatives

These names were chosen after rejecting several alternatives with
terminology conflicts in the target domain:

| Concept | Chosen | Rejected alternatives | Reason for rejection |
|---|---|---|---|
| Per-element callable | **Op** | Stage (RK-stage conflict in every astrophysics code), Kernel (CUDA kernel conflict), Func (Halide, unfamiliar in scientific computing) | "Stage" is universally used for RK substages in Athena++, Castro, Parthenon; "Kernel" already means the dispatch unit in GPU programming |
| Iteration extent | **Region** | Domain (simulation domain conflict), Partition (implies a piece of something; doesn't cover full-domain cases), Pack (Parthenon-specific; implies batching which is a property, not the concept itself), IndexSet (accurate but too low-level for meshblocks and particle collections) | "Domain" in astrophysics means the simulation volume; "Region" is used in Legion and AMReX without the simulation-domain connotation |
| Execution organization | **Policy** | ExecutionPolicy (verbose), Schedule (Halide schedule conflates algorithm + execution), Strategy (non-standard) | Follows Kokkos's established `RangePolicy` / `TeamPolicy` naming; "Schedule" in Halide means more than just execution organization |
| Dispatch unit | **Dispatch** | Sweep (operator-split direction-sweep conflict), Kernel (CUDA kernel conflict), Pass (compiler/render-pipeline jargon), Loop (implies CPU) | "Sweep" in astrophysics means iterating in one spatial direction (x-sweep, y-sweep); "Kernel" already means the dispatch unit in GPU programming; `Dispatch(op, region, policy)` reads more directly than `Pass(op, region, policy)` in Python code |

The alternatives in the Op column ("Stage", "Func") and the Region
column ("Partition") remain worth reconsidering as the interface
solidifies against real workloads in Epoch 1.

## Consequences

- **Positive.** The three axes are independently tunable without
  touching physics code. Fusion experiments require only changes to
  Dispatch composition in the driver. Execution policy can be upgraded
  from flat to tiled to warp-specialized without modifying Ops.
  The naming is unambiguous within the astrophysics simulation
  context.
- **Negative.** Four named concepts where the prior ADR-0002 language
  implied one ("descriptor") adds conceptual overhead for new
  contributors. "Dispatch" and "Op" are non-standard in astrophysics
  simulation codebases; onboarding documentation must introduce them.
- **Neutral.** Only `FlatPolicy` is implemented in Epoch 1; the other
  policies and their backend adapters are deferred. The abstraction
  boundary is established now so that adding them does not require
  restructuring physics code.

## Implementation staging

- **Epoch 1:** `Op` declaration with `access_pattern` metadata
  (`Stencil` only); `Region` with single-block and batched variants;
  `FlatPolicy`; `Dispatch` as the dispatch unit. JAX primary backend
  only. Under `FlatPolicy`, Field arguments to `__call__` may lower to
  raw JAX arrays with Region-coordinate indexing over the local payload.
- **Epoch 2–3:** `TiledPolicy` when mesh stencil operations are
  written. Introduce `FieldView` as a JAX-pytree wrapper that
  presents Region-coordinate semantics over a halo-extended SRAM tile,
  enabling `TiledPolicy` to swap in without changes to any Op.
  Halo-fill coordination between the task graph and the Region's
  declared footprint.
- **Epoch 6:** `WarpSpecializedPolicy` for nuclear reaction networks,
  if benchmarking shows `FlatPolicy` is insufficient. May require
  a code-generation path (SymPy → Triton / Pallas) rather than a
  pure library implementation.

## Resolved design questions

The following design questions were resolved while this ADR was still
Proposed, before the Epoch 1 implementation PR was opened:

1. **`__call__` signature — resolved.** Option A: element indices
   plus an array-like field argument supporting `__getitem__` with
   Region-coordinate semantics. FieldSegments record payload Extents,
   Placement records SegmentId process/device ownership, and Dispatch
   validates coverage before lowering. `TiledPolicy` passes a
   `FieldView` wrapper that redirects Region-coordinate accesses to a
   halo-extended SRAM tile; the Op code is unchanged across policies.
   See Op section above.

2. **Non-stencil access patterns — deferred.** `access_pattern:
   AccessPattern` replaces the narrower `stencil: Stencil` attribute,
   but Epoch 1 only defines the `Stencil` concrete subtype required by
   the Laplacian benchmark. Particle gather/scatter and diagnostic
   reductions are anticipated extensions, not part of the Epoch 1 API.
   They must not be added as thin `radius` or `accum` wrappers: scatter
   needs read/write footprints and write-conflict semantics, while
   reductions need identity, combiner, result location, result
   shape/dtype, scope, finalization, and deterministic/distributed
   reduction expectations. See *AccessPattern* section above.

3. **Policy taxonomy completeness — resolved.** Reductions do not add
   a public `ReducePolicy`. They change output assembly and require
   algebraic metadata (identity, combiner, result location, and
   optional finalization), not a separate author-facing execution
   organization. Block-local, tree, warp, or multi-stage reduction
   implementations are backend lowering strategies inside `FlatPolicy`,
   `TiledPolicy`, or a future Policy.

4. **Naming in code — resolved, revisit after implementation.** `Op`
   remains the computational-axis concept, but subclassing `Op` is no
   longer mandatory; simple operations may be function-shaped via
   `@op(...)`. `Region` is retained for the spatial/iteration extent
   because alternatives such as `Domain`, `Subdomain`, `Block`,
   `Pack`, `Extent`, and `IndexSet` are more overloaded or more
   implementation-specific. The dispatch unit is named `Dispatch`,
   replacing `Pass`, because `Dispatch(op, region, policy)` reads more
   directly in Python code while preserving the concept: one scheduled
   launch boundary over a Region under a Policy. Revisit `Dispatch`
   after the first implementation PR if real call sites show that it
   reads poorly or collides with driver terminology.

5. **Epoch 1 testing path — resolved.** Physics authors should verify
   Ops by constructing and executing a real `Dispatch`, for example
   `Dispatch(op, region, policy=FlatPolicy(), inputs=(field,)).execute()`.
   Direct element-level `op(...)` calls remain useful for small
   pure-function checks, but they are not sufficient as the default
   because they bypass `Dispatch` construction, access-pattern
   validation, Region iteration, future output assembly metadata, and
   backend tracing. Epoch 1 should not add a separate `run(...)` helper
   unless repeated real call sites show that the direct `Dispatch` API
   creates unnecessary duplication.

## Alternatives considered

**Keep Stage / Sweep / Domain** (prior terminology). Rejected because
"Stage" conflicts with Runge-Kutta stage terminology used in every
reference code, "Sweep" conflicts with direction-sweep terminology in
operator-split hydro, and "Domain" conflicts with the simulation domain
concept. The naming confusion would be a persistent maintenance burden
in documentation and onboarding.

**Adopt Kokkos terminology directly** (`parallel_for`, `TeamPolicy`,
`View`). Rejected because Kokkos names are tied to C++ template
instantiation semantics; importing them into a Python API would imply
a compilation model that does not apply here, and `parallel_for` is a
verb phrase rather than a named concept.

**Halide algorithm / schedule separation.** The Halide split is the
closest intellectual precedent for separating computation from
execution organization. Rejected as primary naming because "schedule"
in Halide encompasses both spatial tiling decisions and execution
decisions that we assign to separate axes (Region vs. Policy); adopting
it would import an ambiguity rather than resolving one.

**Require all Ops to subclass `Op`.** Rejected as too nominal for the
long-term Python API. Subclassing is useful for parameterized
operations, stable named physics objects, helper methods, and
constructor-time specialization, so the reference API still provides
an optional `Op` abstract base class. But the backend and Dispatch assembly
only require a callable with metadata. Making inheritance mandatory
would force generated functions, test-local kernels, simple source
terms, and plugin-provided callables through one object model without
adding execution-time guarantees. The public contract is therefore
structural: function-shaped Ops use `@op(...)`, class-shaped Ops may
subclass `Op`, and `Dispatch` accepts any object satisfying the
protocol.

## Cross-references

- `research/01-frameworks.md` §1.2 (Kokkos hierarchical execution
  model — the primary design reference for Policy).
- `research/01-frameworks.md` §1.6 (Singe — motivates
  `WarpSpecializedPolicy` and the algorithm / execution separation).
- `research/07-implications.md` §7a item 2 (kernel abstraction
  implications for the engine's core infrastructure layer).
- [`roadmap/epoch-01-kernels.md`](../roadmap/epoch-01-kernels.md)
  (implementation constraints and exit criterion for Epoch 1).
- ADR-0002 (JAX primary backend) — amended 2026-04-15 to reference
  this ADR for the Dispatch / Op / Region / Policy vocabulary.
- ADR-0001 (Python-only engine, runtime codegen) — `WarpSpecializedPolicy`
  will likely require the codegen path described there.
