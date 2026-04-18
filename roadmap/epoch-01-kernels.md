# Epoch 1 — Kernel abstraction and multi-backend core

> Part of the [Cosmic Foundry roadmap](index.md).

## Required reading

Before finalizing the kernel interface, read:

- Grete et al. (2023), *Parthenon — a performance portable
  block-structured AMR framework*, IJHPCA 37(5), 465. arXiv:
  [2202.12309](https://arxiv.org/abs/2202.12309) — specifically the
  `MeshBlockPack` and `TaskList` designs, which are the closest
  clean-room-readable precedents for the Region batching and Policy
  concepts in ADR-0010.
- Bauer, Treichler, Aiken (2014), *Singe: Leveraging Warp
  Specialization for High Performance on GPUs*, PPoPP 2014.
  PDF: <https://cs.stanford.edu/~sjt/pubs/ppopp14.pdf> — motivates
  the warp-specialized Policy and the principle that execution
  organization must be separable from computation description.

## What this epoch delivers

Four pieces of infrastructure that every later epoch assumes:

- **Kernel interface** — an `Op` / `Region` / `Policy` / `Dispatch`
  abstraction layer (ADR-0010) plus a JAX adapter implementing
  `FlatPolicy` only. Secondary-backend adapters (Numba, Taichi, Warp,
  Triton) remain stubbed extras per ADR-0002.
- **Field placement** — a `Field` data object composed from one or more
  `FieldSegment` payloads plus explicit `Placement` metadata built on
  `jax.distributed` (NCCL / GLOO per ADR-0003) for cross-device /
  cross-host cases. A Field is not assumed to be globally complete;
  Region and Dispatch determine what extent a given operation requires.
- **HDF5 I/O** via `h5py`, parallel where `jax.distributed` can be
  composed with a parallel-HDF5 build, otherwise per-process writes
  with a post-processing merge step.
- **Deterministic structured logging** and error-handling conventions.

## Implementation sequence

Epoch 1 should land as a sequence of small, independently reviewable
PRs. The ordering below keeps correctness checks close to each new
concept and avoids pulling Epoch 2 driver responsibilities into the
kernel layer.

1. **Kernel interface nucleus.** Implement `AccessPattern` /
   `Stencil`, the `@op(...)` decorator, the optional `Op` ABC,
   single-block `Region`, `FlatPolicy`, and direct
   `Dispatch(op, region, policy=FlatPolicy(), inputs=(field,)).execute()`
   execution. The first executable workload is a 3-D 7-point Laplacian
   over one in-memory JAX array using that public Dispatch API.
2. **Region batching.** Add the batched-Region path and lower it with
   JAX `vmap`, proving that the same Laplacian Op runs unchanged over
   one block or a packed collection of at least two blocks.
3. **Field placement smoke path.** Introduce the smallest Field /
   FieldSegment / Placement model needed for the Laplacian exit
   criterion: a FieldSegment payload, SegmentId, Extent metadata,
   process/device ownership, a single-process placement whose one
   segment covers the whole test Region, a two-process placement whose
   segments cover disjoint extents, and a `jax.distributed`
   correctness harness. This is not the Epoch 2 mesh hierarchy and
   should not grow AMR, task-graph, ghost-fill, or refinement-boundary
   behavior.
4. **I/O and observability.** Add HDF5 output for the Laplacian result,
   using parallel HDF5 when available and the per-rank-write /
   post-processing merge pattern otherwise. Wire deterministic
   structured logging around dispatch, sharding, and I/O boundaries.
5. **Benchmark and visual artifact.** Add a CPU roofline benchmark for
   the Dispatch path using a pointwise triad, optional GPU numbers when
   a runner exists, and the first reference render of one Laplacian
   slice under the house colormap.

Secondary backend adapters, non-stencil access patterns, `TiledPolicy`,
`WarpSpecializedPolicy`, AMR, task-graph scheduling, and production
halo-exchange semantics are explicit non-goals for Epoch 1 unless a
later ADR changes the boundary.

## Field placement model

Epoch 1 uses `Field`, not `ShardedField`, as the data-bearing concept.
The design deliberately avoids assuming that every Field is a globally
complete simulation-domain object. A Field is a collection of
FieldSegments. Each FieldSegment pairs a payload with the Extent over
which that payload is valid. Placement maps SegmentIds to
process/device owners. Region and Dispatch decide whether a Field's
placed segments cover the operation being requested.

```text
Concept        Owns                              Does not own
Field          semantic label, segment set       iteration extent, process topology
FieldSegment   payload, Extent                   process/device ownership
Placement      SegmentId -> owner map            physical meaning, kernel lowering
Region         iteration coordinates             storage, ownership, communication
Dispatch       Field/Region/Policy binding       global task ordering, ghost fills
```

Single-process execution is the degenerate case: one Field has one
FieldSegment whose SegmentId is owned by the only process in Placement
and whose Extent covers the whole Region needed by the Dispatch.
Multi-process execution uses the same API with multiple FieldSegments
whose SegmentIds map to different process owners and whose Extents
cover disjoint Region extents. Dispatch validation is responsible for
checking that each Field's placed segments cover the Region plus the
Op's access footprint before lowering. Epoch 2 may extend this model
with meshblocks, centering, ghost ownership, AMR levels, and task-graph
communication, but those are not part of the Epoch 1 Field contract.

## Design constraints for the kernel interface

See ADR-0010 for the full rationale. The summary relevant to
implementation is below.

### Three independent axes

The research survey (§1.2 Kokkos, §1.6 Singe in
`research/01-frameworks.md`) establishes that performance-portable
kernel abstraction requires three independently variable axes:

**Computational axis — Op.** Any per-element callable satisfying the
Op protocol: it declares an `access_pattern` attribute and implements
traceable call behavior. Examples: Helmholtz EOS, HLLC Riemann solver,
PPM reconstruction, Laplacian stencil, CFL computation. An Op is not
dispatchable on its own. Its call behavior must be traceable in the
backend's execution context (JAX-jittable for the primary backend).
The `access_pattern` attribute is used by the driver to derive halo
sizes. Epoch 1 defines `Stencil` only; particle gather/scatter and
diagnostic reductions are anticipated extensions whose metadata is not
frozen until those workloads arrive. The reference API provides both
an `@op(...)` decorator for function-shaped Ops and an optional `Op`
abstract base class for class-based or parameterized Ops. See ADR-0010
for the full interface, `AccessPattern` scope, and metadata catalog.

**Spatial / iteration axis — Region.** The set of elements over which
a Dispatch iterates. May represent a single meshblock, a face array, a
particle array, or a packed collection of meshblocks (analogous to
Parthenon's `MeshBlockPack`). Region batching — packing multiple
meshblocks into one contiguous allocation so a single kernel launch
covers all of them — is a concern of the Region, not of the Op or the
Policy. In JAX, batching over a stacked Region maps to `vmap`; the Op
is unaware of the batch dimension.

**Execution axis — Policy.** How threads are organized to process a
Region. Three policies are anticipated, implemented incrementally:

| Policy | Thread organization | Memory used | Analogous to |
|---|---|---|---|
| `FlatPolicy` | One thread per element, same code for all | HBM only | Kokkos `RangePolicy` |
| `TiledPolicy` | Thread team; cooperative scratchpad load + barrier | HBM + team SRAM | Kokkos `TeamPolicy` + `team_scratch` |
| `WarpSpecializedPolicy` | Warps assigned to different code paths within a block | HBM + role-partitioned registers | Singe warp specialization |

Only `FlatPolicy` is implemented in Epoch 1. `TiledPolicy` is
anticipated in Epoch 2–3 when mesh stencil operations become real
workloads. `WarpSpecializedPolicy` is anticipated in Epoch 7 when
nuclear reaction networks require it. The interface must be designed
so that swapping policies requires no changes to Op implementations.

### The Dispatch

A **Dispatch** is the dispatch unit: one or more Ops applied to a
Region under a Policy. One Dispatch equals one kernel launch (one
`jit` boundary in JAX). The physics author writes Ops. The driver
(Epoch 2) assembles Ops into Dispatches with an attached Policy.
Fusion experiments — grouping or splitting operations across kernel
launches — are answered by changing how Ops are composed into
Dispatches, not by touching Op implementations.

### What is not part of the kernel interface

The **task graph** — dependency tracking across Dispatches, communication
overlap, AMR refinement-boundary synchronization, dynamic load
balancing — belongs in the Epoch 2 driver. The task graph and Region
batching solve different problems (latency hiding vs. launch
amortization) and must remain in separate layers. Epoch 1's kernel
interface is stateless with respect to task ordering. `reads` /
`writes` metadata is authoring shorthand; the Epoch 2 driver normalizes
it with the Dispatch Region and AccessPattern into dependencies over a
field, iteration extent, access mode, and expanded access footprint.
Data dependencies between Dispatches order work but do not necessarily
prohibit a future backend from transparently fusing compatible
Dispatches. Explicit no-fusion/materialization boundaries are
driver/task-graph fences, needed for communication, AMR
synchronization, host-visible diagnostics or I/O, externally consumed
reductions, profiling/timing boundaries, and any intermediate field
that must be materialized before later work.

## Exit criterion

A 3-D 7-point Laplacian implemented as an Op under `FlatPolicy`:

- Runs over a single-meshblock Region and a batched Region (at least
  two meshblocks packed), demonstrating that Region batching works
  without changes to the Op.
- Provides a CPU roofline sanity benchmark for the Dispatch path using
  a pointwise triad; GPU numbers are added when a GPU runner is
  available.
- A multi-rank correctness test verifies that the Field placement
  Laplacian matches the single-rank result.
- A reference render of one benchmark slice under the house colormap is
  committed as the first production visual-regression artifact.
