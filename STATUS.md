# Cosmic Foundry — Status

The immediate implementation queue. Items belong here when they are fully
specified and unblocked — direct line-of-sight on what to implement. Items
not yet specified well enough belong in [`ROADMAP.md`](ROADMAP.md).

For the repository layout, see [`README.md`](README.md).
For architectural decisions, see [`ARCHITECTURE.md`](ARCHITECTURE.md).
For development workflow, see [`DEVELOPMENT.md`](DEVELOPMENT.md).
For the long-horizon capability sequence, see [`ROADMAP.md`](ROADMAP.md).

---

## Completed

**[PR #181] Generalize stencil generation to support any approximation order.**
✅ Extracted parameterizable stencil derivation into `stencil.py` via `derive_laplacian_stencil(order, ndim)` using SymPy `finite_diff_weights`, supporting arbitrary approximation orders (2, 4, 6, ...). Moved codegen pattern (`_derive()`, `generate()`, generated block) from `laplacian.py` into `stencil.py`, following the convention that codegen lives alongside what it produces. Deleted `laplacian.py`; updated all importers (4 files + generator script). Added comprehensive tests in `test_stencil_derive.py` for orders 2 and 4 in 1D and 3D; verified weights-sum-to-zero invariant. Generated block output for order=2 is bit-for-bit identical to original (drift test confirms). No regressions; all tests pass.

---

## Current work

**Implement auto-discovery for kernel module generation and testing.**
With the parameterizable stencil derivation proven in `stencil.py`, the next
step is to generalize the generator script and test suite to auto-discover and
process all kernel modules in one pass:

1. Update `scripts/generate_kernels.py` to auto-discover all modules in
   `cosmic_foundry/computation/` that expose a `generate()` function (skip
   private `_*.py` modules), then splice their generated blocks in
   deterministic order (alphabetical by module name for reproducibility).
2. Update `tests/test_generated_kernels.py` to use the same discovery logic,
   building a parameterized pytest test that verifies each kernel module's
   generated block matches its derivation (drift check). Each module's test
   runs independently, named `test_<module>_constants_match_derivation`.

Once auto-discovery is in place, the framework scales: each new kernel module
only needs to implement `_derive()` and `generate()` with no generator/test
boilerplate to copy.

**Scale derivation pattern to the full codebase.**
After auto-discovery is complete, audit every existing operator
(`cosmic_foundry/computation/*.py` excluding `_codegen.py`) and apply the
derivation pattern: add `_derive()` that returns constants and stencil
structure, and `generate()` that emits the full block. The parameterized test
will automatically verify each one. Any operator without a `_derive()` is not
yet compliant with architectural basis claim 5 (every numerical method formally
derived).

Start with operators that have hardcoded constants (stencils, reductions) where
the pattern applies immediately. Operators without a formal derivation (e.g.,
field sampling, overlap operations) may require a minimal `_derive()` that
documents why derivation is not applicable.

**Remove generated blocks from `stencil.py`; adopt parameterizable-only API.**
Once the full codebase is using the derivation pattern, eliminate named
pre-instantiated objects from `stencil.py` (currently `seven_point_laplacian`).
`stencil.py` should export only the parameterizable generator
`derive_laplacian_stencil(order, ndim)` — the single source of truth. Callers
invoke it directly: `derive_laplacian_stencil(2, 3)` returns the order-2 3D
stencil. Remove `_derive()`, `generate()`, and the generated block; migrate the
drift test to verify `derive_laplacian_stencil(2, 3)` weights match the
derivation. Update all importers (tests, benchmarks) to call the function
instead of importing a named instance. This eliminates namespace pollution and
makes the API scale cleanly as new orders and operators are added.
