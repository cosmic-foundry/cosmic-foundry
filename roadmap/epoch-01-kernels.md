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
- **`ShardedField`** — a distributed-array primitive built on
  `jax.distributed` (NCCL / GLOO per ADR-0003) for cross-device /
  cross-host cases.
- **HDF5 I/O** via `h5py`, parallel where `jax.distributed` can be
  composed with a parallel-HDF5 build, otherwise per-process writes
  with a post-processing merge step.
- **Deterministic structured logging** and error-handling conventions.

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
The `access_pattern` attribute (a `Stencil`, `GatherScatter`,
`Reduction`, or `Composite` thereof) is used by the driver to derive
halo sizes and by the Policy to determine output assembly (element
field vs. reduced scalar). The reference API provides both an
`@op(...)` decorator for function-shaped Ops and an optional `Op`
abstract base class for class-based or parameterized Ops. See ADR-0010
for the full interface, `AccessPattern` type hierarchy, and metadata
catalog.

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
workloads. `WarpSpecializedPolicy` is anticipated in Epoch 6 when
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
- Produces documented roofline fractions on CPU; GPU numbers added
  when a GPU runner is available.
- A multi-rank correctness test verifies that a `ShardedField` Laplacian
  matches the single-rank result.
- A reference render of one benchmark slice under the house colormap is
  committed as the first production visual-regression artifact.
