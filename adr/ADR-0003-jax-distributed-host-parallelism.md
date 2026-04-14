# ADR-0003 — jax.distributed + NCCL as the host-parallelism baseline

- **Status:** Accepted
- **Date:** 2026-04-14

## Context

ADR-0002 commits the primary kernel backend to JAX + XLA, whose
`pjit` / `shard_map` primitives provide device parallelism within
the set of devices JAX can address. That covers multi-GPU on a
single node natively, but it does not answer how multiple nodes are
composed — a decision that controls the `ShardedField` interface
landing in Epoch 1, the I/O path, and the deployment shape of every
physics run.

Two candidate transports:

1. **MPI as the between-node transport**, consumed via `mpi4py`
   and composed with JAX inside the node via `mpi4jax` or
   `jax.distributed`'s MPI backend. This is the pattern every
   reference engine in RESEARCH.md §6 uses (AMReX, Parthenon,
   AthenaK, SWIFT, Flash-X, GADGET-4, Arepo). Broadest HPC-site
   coverage; oldest investment base.
2. **`jax.distributed` + a modern collective (NCCL for GPU,
   GLOO for CPU)** as the primary transport, with MPI available as
   a per-site fallback when `jax.distributed` cannot initialize.
   This is the pattern large-scale ML and the ML-adjacent scientific
   workloads increasingly deploy on.

The choice is a strategic bet on where the field is going. ML
infrastructure investment — NCCL, NVSHMEM, GLOO, TPU-pod collectives
— has converged on a different set of primitives than classical HPC
MPI stacks. Several consequences follow from picking one or the
other:

- **MPI-primary** buys immediate reach into every major HPC facility
  and aligns with reference-engine scaling experience, at the cost
  of carrying a second layer of parallelism alongside JAX, plus the
  known-rough composition of MPI + CUDA-aware transports when sites
  disagree on MPI / CUDA build conventions.
- **`jax.distributed`-primary** keeps the stack single-layered and
  aligned with modern ML-infra investment, at the cost of deployment
  risk: some HPC sites still lack a turnkey NCCL or GLOO path over
  their native interconnects, and a per-site porting effort may be
  required to run at facility scale.

Cosmic Foundry's trajectory is to *replicate* the capabilities of
the reference engines rather than to run inside their facility
stacks. That framing makes the modern-infra bet more palatable: the
engine does not need to deploy on every HPC site on day one; it
needs a clean infrastructure layer that compounds in value as ML
and scientific collective primitives continue to converge. A per-
site MPI path remains available if a future facility deployment
requires it.

## Decision

Host parallelism is delivered by **`jax.distributed`** using
**NCCL** as the GPU collective and the standard `jax.distributed`
CPU path (GLOO) as the CPU collective. No MPI layer is in the
baseline.

- Epoch 1's `ShardedField` is implemented against `jax.distributed`
  semantics. `pjit` / `shard_map` continue to provide within-node
  device parallelism, composed with `jax.distributed` for between-
  host collectives, producing a single-layer programming model.
- Parallel I/O is done through HDF5, either via `h5py` built against
  parallel HDF5 when available or via a per-rank-write + post-
  processing merge pattern when it is not. The I/O path does not
  assume MPI.
- `mpi4py` and `mpi4jax` are **not** included in the baseline
  dependencies. They remain available as optional extras for sites
  where `jax.distributed` cannot initialize over the native
  interconnect. Adopting MPI as a required transport for any epoch
  in the baseline would require a new ADR superseding this one.
- The `cosmic-foundry hello` entrypoint in Epoch 0 reports
  `jax.distributed` state (`process_index`, `process_count`,
  local / global device lists) instead of `mpi4py` rank / size.
- Multi-host CI is scaffolded in Epoch 0 behind a `--multihost`
  pytest marker (not run) and turned on in Epoch 1 once a
  `ShardedField` smoke test exists on a two-process `jax.distributed`
  harness.

## Consequences

- **Positive.** One layer of parallelism rather than two; the
  `ShardedField` contract, collective semantics, and debugging tools
  all live under a single API. The engine tracks the trajectory of
  ML infrastructure investment (NCCL, NVSHMEM, TPU-pod collectives)
  rather than anchoring to a transport whose investment base is
  concentrated at HPC facilities. Autodiff and collective
  communication compose natively within `jax.distributed` — no
  second interop surface.
- **Negative.** Deployment risk at classical HPC facilities that
  lack a turnkey NCCL path over their native interconnect; each
  such facility requires a porting investment or the optional MPI
  path. Reference-engine scaling experience (thousands of MPI
  ranks) does not transfer directly — scaling data must be rebuilt
  on the new stack. Bitwise reproducibility holds within a JAX /
  XLA version on a fixed device; collective reductions over floats
  are not associative, so cross-topology bitwise reproducibility is
  not claimed.
- **Neutral.** `roadmap/index.md` §2 currently lists `mpi4py`,
  `mpi4jax`, and parallel HDF5 as baseline dependencies — a
  follow-up roadmap edit is required to align §2 with this ADR.
  `environment/cosmic_foundry.yml` includes `openmpi` as a conda
  dependency; this becomes optional under the revised baseline and
  can be dropped in a later cleanup PR. The §0.7 `cosmic-foundry
  hello` description in `roadmap/epoch-00-bootstrap.md` also refers
  to `MPI.Init()` and will need a small edit to match the
  `jax.distributed` path.

## Alternatives considered

- **MPI-between-nodes + JAX-within-node.** The classical HPC
  pattern; broadest facility coverage today. Rejected as the
  baseline because it doubles the parallelism stack for a benefit
  (immediate turnkey deployment at every HPC site) that the
  project's trajectory does not need to capture on day one. Remains
  available as an optional path.
- **Charm++ / Legion / UPC++ task-based runtimes.** Architecturally
  attractive and the scaling story behind SWIFT, Cactus+Carpet,
  and Parthenon. None exposes a native Python API at the maturity
  `jax.distributed` does today; adopting one would require a second
  interop surface that competes with the runtime-codegen bet in
  ADR-0001. Revisitable via a dedicated ADR if `jax.distributed`
  hits a wall.
- **Defer host parallelism to Epoch 2 or later.** Would keep
  Epoch 1 smaller but means `ShardedField` is designed against
  single-host semantics and retrofitted with collectives later —
  exactly the refactor a host-parallelism-first decision is trying
  to prevent.

## Cross-references

- [`roadmap/index.md`](../roadmap/index.md) §2 (Technology baseline
  — requires a follow-up edit to align with this ADR), §5
  (crossroads: host-parallelism model, now resolved).
- [`roadmap/epoch-00-bootstrap.md`](../roadmap/epoch-00-bootstrap.md)
  §0.7 (`cosmic-foundry hello` — requires a follow-up edit to replace
  `mpi4py.MPI.Init()` with `jax.distributed.initialize()`).
- [`roadmap/epoch-01-kernels.md`](../roadmap/epoch-01-kernels.md)
  (Epoch 1 `ShardedField` and multi-host CI).
- ADR-0002 (JAX primary backend) — composes with this decision for
  intra-node parallelism.
