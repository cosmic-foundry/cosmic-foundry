# ADR-0003 — MPI in the parallelism baseline from Epoch 1

- **Status:** Accepted
- **Date:** 2026-04-14

## Context

ADR-0002 commits the primary kernel backend to JAX + XLA, whose
`pjit` / `shard_map` primitives provide device- and host-parallelism
*within* the set of devices JAX can see. That covers multi-GPU on
a single node and the TPU-pod shape natively, but it does not
address the multi-node HPC deployments every reference engine in
RESEARCH.md §6 targets (AMReX, Parthenon, AthenaK, SWIFT, Flash-X,
GADGET-4, Arepo, …), which all run on MPI.

The roadmap's Epoch 1 ([`roadmap/epoch-01-kernels.md`](../roadmap/epoch-01-kernels.md))
delivers the first multi-rank field primitive (`ShardedField`), so
the question "is MPI in the baseline, or is it deferred?" must be
answered before Epoch 1 opens, not after.

Two candidate approaches:

1. **Pure JAX host-parallelism** (`jax.distributed` over a
   collective like NCCL / GLOO). Keeps the stack single-layer, but
   requires every HPC site we target to have a JAX-compatible
   transport available, which is not yet universal on
   supercomputing facilities.
2. **MPI as the between-node transport, JAX inside the node.**
   Matches how every listed reference engine deploys. Requires
   mpi4py + mpi4jax or `jax.distributed` wired to use MPI, plus
   the usual parallel-HDF5 pattern for I/O.

The MPI-between-nodes path is directly supported by `mpi4jax` and by
`jax.distributed` with an MPI backend, and it is the only path that
works unmodified on every HPC site Cosmic Foundry is plausibly
deployed on. Committing to it in the baseline is cheaper than
retrofitting MPI through mature physics modules once the engine
leaves the workstation-GPU phase.

## Decision

MPI is in the parallelism baseline from Epoch 1.

- **Between-node transport:** MPI, consumed from Python via
  `mpi4py`.
- **Device / within-node parallelism:** JAX `pjit` / `shard_map`
  (per ADR-0002), composed with MPI via `mpi4jax` and / or
  `jax.distributed`'s MPI backend, whichever proves more robust
  under the Epoch 1 `ShardedField` tests.
- **Parallel I/O:** `h5py` built against parallel HDF5, with a
  collective-write pattern for checkpoints and plotfiles.
- **Multi-rank CI:** scaffolded in Epoch 0 behind a `--mpi` pytest
  marker (not run), turned on in Epoch 1 as a dedicated job once
  the first `ShardedField` test exists.

The `cosmic-foundry hello` entrypoint shipped in Epoch 0 already
calls `mpi4py.MPI.Init()` and prints rank / size, so the MPI path
is exercised end-to-end before any physics lands.

## Consequences

- **Positive.** Deployment targets the full range of HPC sites
  without per-site porting. Reference-engine scaling experience
  (strong / weak scaling at thousands of ranks) transfers directly.
  The `ShardedField` contract is expressible in terms well-
  understood by every reference code's users.
- **Negative.** Two layers of parallelism (MPI + JAX device)
  instead of one; composition has known rough edges (collective
  ordering, device placement on MPI spawn, CUDA-aware MPI
  availability varying by site). Debugging multi-rank failures is
  harder than single-process JAX.
- **Neutral.** `mpi4jax` is a small additional dependency;
  `jax.distributed` with MPI is a pure JAX code path. Either choice
  is reversible as long as the interface against `ShardedField` is
  kept narrow.

## Alternatives considered

- **`jax.distributed` with a non-MPI collective (NCCL / GLOO)
  only.** Simplest stack, but not universally deployable to the
  supercomputing sites that drive Cosmic Foundry's reference set.
  Rejected as a deployment-risk decision, not a technical one.
- **Charm++ / Legion / UPC++ as the host-parallelism model.** Task-
  based runtimes are architecturally attractive and are how SWIFT,
  Cactus+Carpet, and Parthenon extract their scaling. None exposes
  a native Python API of the maturity JAX + MPI does today;
  adopting one would require a second interop layer that competes
  with the runtime-codegen bet in ADR-0001. Revisitable later via
  a dedicated ADR if the JAX + MPI composition hits a wall.
- **Defer MPI to Epoch 2 or later.** Would keep Epoch 1 smaller,
  but retrofitting multi-node parallelism after the first
  `ShardedField` test passes on single-rank hardware is the exact
  refactor MPI-first is trying to prevent. The Epoch 1 cost of
  shipping `ShardedField` with MPI already wired is smaller than
  the cost of re-plumbing it later.

## Cross-references

- [`roadmap/index.md`](../roadmap/index.md) §2, §5 (crossroads: host
  parallelism model).
- [`roadmap/epoch-00-bootstrap.md`](../roadmap/epoch-00-bootstrap.md) §0.4, §0.7.
- [`roadmap/epoch-01-kernels.md`](../roadmap/epoch-01-kernels.md)
  (Epoch 1 `ShardedField` and multi-rank CI).
- ADR-0002 (JAX primary backend) — composes with this decision for
  intra-node parallelism.
