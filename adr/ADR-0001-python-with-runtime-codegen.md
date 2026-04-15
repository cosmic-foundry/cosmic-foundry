# ADR-0001 — Python-only engine with runtime code generation

- **Status:** Accepted
- **Date:** 2026-04-14

## Context

Cosmic Foundry's goal (RESEARCH.md §6–§7,
[`roadmap/index.md`](../roadmap/index.md) §1) is a fully self-contained
engine covering the union of capabilities cataloged in RESEARCH.md §6.
Every published superset of those capabilities today — AMReX / Parthenon /
Kokkos / Charm++ family, Flash-X, AthenaK, SWIFT, Arepo, MESA — is
built on a compiled C++ or Fortran core with thin Python bindings for
setup and analysis.

A fresh implementation has to decide, at the root, whether to follow
that architecture or to invert it: move the whole engine into Python
and produce any native code the engine needs at runtime rather than
shipping compiled extensions from this repository. That choice
propagates into every other infrastructural decision: kernel
interface, build system, packaging, CI, test strategy, and contributor
ramp-up cost.

Two properties push the decision toward Python:

- **Single language through the whole stack.** Problem setup, driver
  logic, physics kernels, I/O, and visualization become one language
  — eliminating the FFI boundary that consumes real engineering
  effort in every C++/Python hybrid engine.
- **Autodiff and symbolic codegen become first-class.** JAX provides
  reverse- and forward-mode autodiff over arrays; SymPy derives
  fluxes, Jacobians, and curvature tensors symbolically, then emits
  Python source that JAX (or another backend) JITs. Hand-derived
  Jacobians — a recurring source of defects in stiff integrators,
  primitive-variable recovery, and BSSN-family evolution terms —
  largely disappear.

The cost is accepting that *performance* must come entirely from
runtime code generation (JAX/XLA, Numba, Taichi, Warp, Triton)
rather than from ahead-of-time compilation of hand-written C++. That
bet is serviceable for two reasons: modern JIT backends hit or
approach hand-tuned performance for the kernels Cosmic Foundry needs,
and all of those backends expose Python APIs, so the engine can stay
in one language while still benefiting from the best of each.

## Decision

Cosmic Foundry is a **Python-only engine**. No compiled extensions
are shipped from this repository. Any native code the engine executes
is produced at runtime by a code-generation backend (see ADR-0002),
driven from Python source and symbolic descriptions.

- Python ≥ 3.11 is the single source language.
- `pybind11` and `ctypes` are tolerated only as emergency escape
  hatches; adopting either requires a new ADR stating which
  performance target forced the exception.
- Ahead-of-time compiled libraries are consumed as pre-built
  dependencies (e.g. JAX, NumPy, h5py) but never produced by this
  repository's build.

## Consequences

- **Positive.** One language end-to-end; no cross-language build
  rules; contributors need only a Python toolchain. Autodiff and
  symbolic codegen are natural rather than bolted on. Packaging is
  a pure `pyproject.toml` with optional extras per backend; wheels
  are not required.
- **Negative.** Performance depends entirely on runtime backends.
  Pathological kernels that do not fit any backend's execution model
  cannot be rescued by dropping to hand-written C++ in-repo. The
  engine's worst-case performance is bounded by the best of
  {JAX/XLA, Numba, Taichi, Warp, Triton}.
- **Neutral.** The approach is empirically novel at this scope: no
  other engine in the RESEARCH.md §6 superset has shipped in pure
  Python with runtime codegen. This ADR commits us to demonstrating
  the bet rather than defending it on precedent.

## Alternatives considered

- **C++ core with Python bindings (AMReX / Parthenon pattern).**
  Known-good, highest-performance, and the most common architecture
  in the reference codes. Rejected because the cross-language
  boundary consumes disproportionate effort at a small-team scale,
  and because autodiff / symbolic codegen over array kernels is
  harder to retrofit into a C++ core than to design around in
  Python.
- **Fortran or Julia as the single source language.** Fortran loses
  the Python scientific-software ecosystem (plotting, notebooks,
  data formats, downstream analysis). Julia offers a cleaner
  performance story but a smaller ecosystem for the adjacent
  capabilities — particularly MPI interop, HDF5, browser-side
  visualization, and community familiarity for contributors.
- **Python with a mandatory compiled extension layer (Cython /
  pybind11).** Hybrid stacks bring back the FFI boundary this
  decision is specifically trying to eliminate, without the
  language-unification benefit of a true C++ core. Rejected as the
  worst of both worlds for this project's scope.

## Cross-references

- [`roadmap/index.md`](../roadmap/index.md) §2 (Technology baseline).
- [`roadmap/epoch-00-bootstrap.md`](../roadmap/epoch-00-bootstrap.md) §0.6.
- RESEARCH.md §7 (Implications for Cosmic Foundry).
- ADR-0002 (JAX as primary kernel backend) — depends on this decision.
