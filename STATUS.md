# Cosmic Foundry — Status

The immediate implementation queue. Items belong here when they are fully
specified and unblocked — direct line-of-sight on what to implement. Items
not yet specified well enough belong in [`ROADMAP.md`](ROADMAP.md).

For the repository layout, see [`README.md`](README.md).
For architectural decisions, see [`ARCHITECTURE.md`](ARCHITECTURE.md).
For development workflow, see [`DEVELOPMENT.md`](DEVELOPMENT.md).
For the long-horizon capability sequence, see [`ROADMAP.md`](ROADMAP.md).

---

## Current work

**Generalize stencil generation to support any approximation order.**
The Laplacian derivation in `laplacian.py` is specific to second-order 1D
stencils extended to 3D. Extract the derivation logic into a parameterizable
stencil generator (e.g., `cosmic_foundry/computation/stencil.py`) that can
produce 1D/3D stencils of any approximation order (2nd, 4th, 6th, etc.).
Move the Laplacian-specific logic into a thin wrapper that calls the generic
generator with order=2. This becomes the foundation for scaling: higher-order
stencils (bi-harmonic, compact schemes) can then be derived and generated
without reimplementing the derivation pattern.

**Implement auto-discovery for kernel module generation and testing.**
`cosmic_foundry/computation/_codegen.py` now provides shared utilities
(sentinels, hash, splice) and the derivation pattern is proven in
`laplacian.py` with full kernel function body generation. The next step is to
generalize the generator script and test suite:

1. Update `scripts/generate_kernels.py` to auto-discover all modules in
   `cosmic_foundry/computation/` that expose a `generate()` function (skip
   private `_*.py` modules), then splice their generated blocks in one pass.
2. Update `tests/test_generated_kernels.py` to use the same discovery logic,
   building a parameterized pytest test that verifies each kernel module's
   generated block matches its derivation (drift check).

Once auto-discovery is in place, the framework is ready to scale: each new
kernel module only needs to implement `_derive()` and `generate()` with no
generator/test boilerplate to copy.

**Scale derivation pattern to the full codebase.**
After auto-discovery is complete, audit every existing operator
(`cosmic_foundry/computation/*.py` excluding `_codegen.py`) and apply the
pattern: add `_derive()` that returns constants and stencil structure, and
`generate()` that emits the full block. The parameterized test will
automatically verify each one. Any operator without a `_derive()` is not yet
compliant with architectural basis claim 5.
