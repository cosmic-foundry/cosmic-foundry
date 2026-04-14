# ADR-0002 — JAX + XLA as the primary kernel backend

- **Status:** Accepted
- **Date:** 2026-04-14

## Context

ADR-0001 commits Cosmic Foundry to runtime code generation: the
engine ships Python source and produces native code at import /
execution time. That decision leaves open the question of *which*
runtime backend is primary — a choice that controls the kernel
interface, the autodiff story, the GPU / TPU portability story, and
the shape of every physics module from Epoch 1 onward.

The RESEARCH §6 capability set rules out any single backend covering
every workload. At least four backends are plausibly needed across
the program's horizon:

- Array-shaped, functional kernels with reverse-mode autodiff and
  multi-device parallelism (hydro, MHD, radiation, NR).
- CPU SIMD / CUDA kernels with more flexible shape and control flow
  than XLA tolerates (certain solvers, indirection-heavy loops).
- Particle / SPH / unstructured / moving-mesh workloads with mature
  simulation-DSL semantics.
- Hand-tuned GPU kernels for the small number of cases that need
  full control.

Committing to multiple backends up-front would multiply design cost
and delay Epoch 1. Committing to one backend risks architectural
lock-in. The resolution is to make one backend *primary* (exercised
through Epoch 0 and Epoch 1) and structure the kernel interface so
secondary backends can slot in later without rewriting callers.

JAX + XLA is chosen as the primary because it alone covers the
four properties the core infrastructure needs simultaneously:
array-level autodiff, JIT compilation targeting CPU / CUDA / ROCm /
TPU through one runtime, `pjit` / `shard_map` for device- and
host-parallelism (combining with mpi4py per ADR-0003), and a mature
Python ecosystem. Numba, Taichi, Warp, and Triton individually cover
a subset but none covers all four.

## Decision

The primary kernel backend is **JAX + XLA**.

- Every kernel shipped through Epoch 1 is authored as a JAX kernel
  and exercised on the JAX/XLA backend.
- A `@kernel` descriptor layer (designed in Epoch 1, sketched as a
  stub in Epoch 0) wraps kernels so that secondary backends can
  register adapters without changing call sites.
- The following secondary backends are listed as optional extras in
  `pyproject.toml` with pinned known-good versions, so the descriptor
  layer has concrete target shapes, but **no adapter is written,
  installed, or exercised in Epoch 0 or Epoch 1**:
  - **Numba** — CPU SIMD and CUDA kernels where XLA's shape /
    control-flow constraints are limiting.
  - **Taichi** — particle / SPH / unstructured / moving-mesh.
  - **NVIDIA Warp** — GPU particle, tree, neighbor kernels; Monte
    Carlo transport.
  - **Triton** — hand-tuned GPU kernels.
- Adopting a second backend in production (as opposed to stubbed
  extras) requires a new ADR stating which workload forced the
  addition and what performance gap the secondary backend closes.

## Consequences

- **Positive.** A single backend to support in Epoch 0–1 cuts scope
  sharply. `pjit` / `shard_map` inside a node composes cleanly with
  mpi4py between nodes (ADR-0003). Autodiff is native, which cashes
  in directly on ADR-0001's symbolic-codegen bet. CPU, CUDA, ROCm,
  and TPU all reachable from one kernel definition.
- **Negative.** XLA constrains shape polymorphism and control flow;
  some solver kernels (indirection-heavy, irregular loops) will
  fit awkwardly until a secondary backend lands. Bitwise
  reproducibility holds within a backend on a fixed device but not
  across backends or across XLA versions.
- **Neutral.** The descriptor layer is additional scaffolding
  introduced for a future that may or may not arrive as specified;
  the design bet is that its cost is small compared to the cost of
  retrofitting multi-backend dispatch later.

## Alternatives considered

- **Numba primary, JAX secondary.** Numba fits indirection-heavy
  kernels better, but its autodiff story is external (via JAX or
  hand-written), and multi-device parallelism is a second system on
  top. Rejected because starting without native autodiff negates
  ADR-0001's central benefit.
- **Taichi primary.** Excellent for particles and unstructured
  meshes, but the array / AMR / physics-kernel ecosystem is thinner
  and autodiff is less mature than JAX's. Rejected for the same
  autodiff reason, and because most of Epoch 1–9 is array- and
  stencil-shaped rather than particle-shaped.
- **No primary, descriptor-first from day one.** Designing the
  descriptor layer against four concrete backends simultaneously
  front-loads interface work before any physics runs. Rejected as
  premature abstraction: Epoch 1 will evolve the descriptor against
  a real JAX kernel workload, then secondary backends are added
  with the interface already pressure-tested.
- **PyTorch.** Excellent autodiff and GPU story, but the ecosystem
  orients around ML workloads; multi-host parallelism semantics,
  MPI interop, and scientific-computing tooling are thinner than
  JAX's for this project's targets.

## Cross-references

- [`roadmap/index.md`](../roadmap/index.md) §2 (Technology baseline).
- [`roadmap/epoch-00-bootstrap.md`](../roadmap/epoch-00-bootstrap.md) §0.2, §0.6.
- [`roadmap/epoch-01-kernels.md`](../roadmap/epoch-01-kernels.md)
  (kernel descriptor, where the interface solidifies).
- ADR-0001 (Python-only engine with runtime codegen) — this ADR depends on it.
- ADR-0003 (MPI baseline) — combines with JAX device parallelism for host parallelism.
