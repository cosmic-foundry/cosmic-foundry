# ADR-0011 — Halo Fill Fence

## Context

`Field.covers(required_extent)` can validate that a local segment contains
the data needed by a `Dispatch`, but there is no mechanism for the driver to
*fill* missing ghost cells before issuing the dispatch.

Epoch 1 sidestepped this by pre-slicing overlapping arrays from a
locally-available full domain.  Production domain decomposition cannot do
that: each rank holds a disjoint slab and must exchange a layer of ghost
cells with its neighbors before applying stencil operations at segment
boundaries.

Two constraints shape the design:

1. **Separation of concerns** — ADR-0010 establishes that computational
   logic (Op), spatial extent (Region), and execution organization (Policy)
   are separate concepts.  Ghost-cell exchange is an execution concern, not
   a computational one; it should not appear inside kernels.

2. **Driver visibility** — The Epoch 2 task-graph driver must be able to
   inspect, reorder, and deduplicate communication before execution.
   Hiding the exchange inside a JAX collective called silently from inside
   `Dispatch.execute` would make it invisible to the scheduler.

## Decision

Introduce a `HaloFillFence` descriptor that records the communication intent
for one field, and a companion `HaloFillPolicy` that carries out the exchange.
The driver is responsible for inserting `HaloFillFence` objects into the task
sequence before any `Dispatch` whose required footprint extends beyond the
local segment interior.

### Concepts

**`HaloFillFence`** is a frozen dataclass:

```python
@dataclass(frozen=True)
class HaloFillFence:
    field: Field
    region: Region
    access_pattern: AccessPattern
```

It answers the question "which halo cells need to be current before this
region can be dispatched?"  It carries no communication logic itself.

**`HaloFillPolicy`** executes the exchange.  The baseline implementation uses
`jax.distributed` point-to-point collectives; the interface is a single
method:

```python
class HaloFillPolicy:
    def execute(self, fence: HaloFillFence, rank: int) -> Field:
        """Return a new Field with ghost cells filled for *rank*."""
```

`execute` returns a new `Field` — a JAX-idiomatic immutable update — rather
than mutating the input.  The caller (driver) replaces the field in the
dispatch sequence with the returned one.

**Required footprint** is computed identically to the existing `covers`
validation:

```python
required = fence.region.extent.expand(fence.access_pattern)
```

Segments on `rank` are identified via `fence.field.local_segments(rank)`.
Segments owned by other ranks are identified via `fence.field.placement`.

### Driver usage pattern

```python
for op_name, bound_op, region in task_sequence:
    field = registry[op_name]
    required = region.extent.expand(bound_op.op.access_pattern)
    if not field.covers(required):
        fence = HaloFillFence(field, region, bound_op.op.access_pattern)
        field = halo_policy.execute(fence, local_rank)
        registry[op_name] = field  # updated segments for subsequent tasks
    results[op_name] = Dispatch(bound_op, region).execute()
```

The driver loop is sketched here to fix the interface contract; the actual
task-graph driver is an Epoch 2 deliverable.

### Information flow

The `HaloFillPolicy` has all the information it needs from existing
primitives:

| What it needs | Where it comes from |
|---|---|
| Halo width per axis | `fence.access_pattern.halo_width(axis)` |
| Local segments and their extents | `fence.field.local_segments(rank)` |
| Which rank owns each neighboring segment | `fence.field.placement.owner(segment_id)` |
| Neighboring segment extents | `fence.field.segment(segment_id).extent` |
| JAX array to pull from | `fence.field.segment(segment_id).payload` |

No new metadata is required.  The `Field`/`Placement`/`FieldSegment` model
already carries everything needed; the fence is purely an orchestration
primitive.

### Non-goals of this ADR

- **Implementation of `HaloFillPolicy`** — that is an Epoch 2 code
  deliverable; this ADR records the interface contract only.
- **Multi-level AMR halo exchange** — AMR introduces coarse-fine boundaries
  where ghost cells are computed by interpolation, not copied.  That is
  deferred to the AMR sub-epoch; this ADR covers uniform-grid exchange only.
- **Batched / tiled regions** — `Region.n_blocks > 1` signals batched
  execution within a rank.  Halos are filled at the field level before
  batching; the fence sees the pre-batched `Field` and a non-batched
  `Region`.

## Consequences

- **Positive:** The driver can inspect the full communication schedule
  before executing anything, enabling reordering, deduplication (two
  dispatches in sequence that need the same halo need only one fence), and
  overlap of communication with independent computation.
- **Positive:** `HaloFillPolicy` is swappable — a stub that asserts coverage
  is already satisfied can be used in single-rank tests; the real
  distributed policy is injected only in multi-rank runs.
- **Positive:** Kernels and `Dispatch` remain communication-free;
  `FlatPolicy.execute` is not modified.
- **Negative:** The driver must track a field registry and update it after
  every `HaloFillFence`; this adds bookkeeping overhead that single-rank
  users who never need halo exchange still pay in complexity.
- **Neutral:** `Field` returns a new object from `HaloFillPolicy.execute`.
  Callers must rebind their registry entry; they cannot cache a reference to
  the original `Field` and expect it to stay current.

## Alternatives considered

**JAX `shard_map` / `with_sharding_constraint`** — Express the distributed
field as a sharded JAX array and let the XLA compiler insert collectives
automatically.  Appealing for single-device-per-rank cases, but:
(a) it requires committing to XLA's sharding model at the field level, which
is higher-level than the `Field`/`Placement` model already in place; (b) the
compiler-inserted collectives are invisible to the driver, precluding explicit
scheduling and deduplication; (c) JAX's sharding model and `jax.distributed`
are not the same thing — `shard_map` operates over device shards within a
process, not across processes.  Deferred to a future ADR if the sharding
model turns out to be the right long-term answer.

**MPI-style explicit `send`/`receive` inside `FlatPolicy`** — Move the
exchange into `FlatPolicy.execute`, triggered when `covers` fails.  This
makes exchange invisible to the driver (same problem as `shard_map`), and
couples the execution policy to a specific communication library.  Ruled out.

**`Field.fill_halos(region, access_pattern)` method** — Add a method
directly to `Field`.  This couples the storage model to the communication
model and violates the separation of concerns established by ADR-0010.
`Field` would need to know about ranks, policies, and network primitives.
Ruled out.

**Single `HaloExchange` object (no policy split)** — A single class that
both describes and executes the exchange, analogous to a self-contained MPI
request object.  Simpler to use but harder to stub, test, and swap
implementations.  The fence/policy split follows the existing Op/Policy
precedent and is preferred.
