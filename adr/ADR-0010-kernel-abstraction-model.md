# ADR-0010 — Kernel abstraction model: Op / Region / Policy / Pass

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

The kernel abstraction layer is structured around four named concepts,
each owning exactly one axis or one composition:

### Op (computational axis)

A per-element callable with a declared stencil footprint. An Op
describes *what* to compute at a single element given its local
neighborhood. It is not dispatchable on its own.

- The stencil footprint is metadata attached to the Op declaration.
  It is used by the driver to determine halo widths for any given
  Region partition size.
- An Op must be expressible in the backend's execution context
  (JAX-jittable for the primary backend; `@njit`-able, `@wp.func`-
  decorated, etc. for secondary backends when they are introduced).
- Examples: Helmholtz EOS evaluation, HLLC Riemann solver, PPM
  reconstruction, 7-point Laplacian stencil, cooling rate per cell.

### Region (spatial axis)

A spatial sub-domain over which an Op is applied. A Region owns two
spatial concerns:

- **Extent**: which elements are iterated over (a meshblock, a face
  array, a particle array).
- **Batching**: whether the Region is a single block or a packed
  collection of blocks (analogous to Parthenon's `MeshBlockPack`).
  Region batching determines how many elements enter a single kernel
  launch; the Op is unaware of the batch dimension.

In JAX, a batched Region maps to `vmap` over the Op; the Op
implementation is unchanged.

### Policy (execution axis)

How threads are organized to process a Region. A Policy is a
swappable object attached to a Pass. Three policies are anticipated:

| Policy | Description | When needed |
|---|---|---|
| `FlatPolicy` | One thread per element; same code for all. Kokkos `RangePolicy` analog. | Default; sufficient for Epochs 1–3. |
| `TiledPolicy` | Thread team with cooperative scratchpad load and explicit barrier. Kokkos `TeamPolicy` + `team_scratch` analog. Addresses bandwidth-bound stencils. | Epoch 2–3 when mesh stencil ops become real workloads. |
| `WarpSpecializedPolicy` | Warps within a thread block assigned to different code paths. Singe analog. Addresses compute-irregular workloads (stiff ODE, tree walks). | Epoch 6 when nuclear reaction networks demand it. |

Swapping a Policy requires no changes to the Op or to the Region. The
physics author does not choose a Policy; the driver or a performance
annotation does.

### Pass (dispatch unit)

A Pass is the composition of an Op, a Region, and a Policy. One Pass
equals one kernel launch — one `jit` boundary in JAX. The Pass is the
unit that the driver schedules, the task graph depends on, and the
profiler attributes.

The physics author writes Ops. The driver assembles Passes. Fusion
experiments are expressed by composing or splitting Ops within a Pass
body, not by modifying Ops.

## Naming rationale and alternatives

These names were chosen after rejecting several alternatives with
terminology conflicts in the target domain:

| Concept | Chosen | Rejected alternatives | Reason for rejection |
|---|---|---|---|
| Per-element callable | **Op** | Stage (RK-stage conflict in every astrophysics code), Kernel (CUDA kernel conflict), Func (Halide, unfamiliar in scientific computing) | "Stage" is universally used for RK substages in Athena++, Castro, Parthenon; "Kernel" already means the dispatch unit in GPU programming |
| Spatial sub-domain | **Region** | Domain (simulation domain conflict), Partition (implies a piece of something; doesn't cover full-domain cases), Pack (Parthenon-specific; implies batching which is a property, not the concept itself) | "Domain" in astrophysics means the simulation volume; "Region" is used in Legion and AMReX without the simulation-domain connotation |
| Execution organization | **Policy** | ExecutionPolicy (verbose), Schedule (Halide schedule conflates algorithm + execution), Strategy (non-standard) | Follows Kokkos's established `RangePolicy` / `TeamPolicy` naming; "Schedule" in Halide means more than just execution organization |
| Dispatch unit | **Pass** | Sweep (operator-split direction-sweep conflict), Kernel (CUDA kernel conflict), Dispatch (verb, not a noun concept), Loop (implies CPU) | "Sweep" in astrophysics means iterating in one spatial direction (x-sweep, y-sweep); "Pass" is used in compilers and GPU render pipelines for "one traversal over data under a computation" |

The alternatives in the Op column ("Stage", "Func") and the Region
column ("Partition") remain worth reconsidering as the interface
solidifies against real workloads in Epoch 1.

## Consequences

- **Positive.** The three axes are independently tunable without
  touching physics code. Fusion experiments require only changes to
  Pass composition in the driver. Execution policy can be upgraded
  from flat to tiled to warp-specialized without modifying Ops.
  The naming is unambiguous within the astrophysics simulation
  context.
- **Negative.** Four named concepts where the prior ADR-0002 language
  implied one ("descriptor") adds conceptual overhead for new
  contributors. "Pass" and "Op" are non-standard in astrophysics
  simulation codebases; onboarding documentation must introduce them.
- **Neutral.** Only `FlatPolicy` is implemented in Epoch 1; the other
  policies and their backend adapters are deferred. The abstraction
  boundary is established now so that adding them does not require
  restructuring physics code.

## Implementation staging

- **Epoch 1:** `Op` declaration with stencil footprint; `Region` with
  single-block and batched variants; `FlatPolicy`; `Pass` as the
  dispatch unit. JAX primary backend only.
- **Epoch 2–3:** `TiledPolicy` when mesh stencil operations are
  written. Halo-fill coordination between the task graph and the
  Region's declared footprint.
- **Epoch 6:** `WarpSpecializedPolicy` for nuclear reaction networks,
  if benchmarking shows `FlatPolicy` is insufficient. May require
  a code-generation path (SymPy → Triton / Pallas) rather than a
  pure library implementation.

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
  this ADR for the Pass / Op / Region / Policy vocabulary.
- ADR-0001 (Python-only engine, runtime codegen) — `WarpSpecializedPolicy`
  will likely require the codegen path described there.
