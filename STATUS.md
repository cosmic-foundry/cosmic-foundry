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

**Clean up `stencil.py`: remove generated-code infrastructure, push to callers.**
Remove the offline code-gen boilerplate from `stencil.py`: delete `_derive()`,
`generate()`, and the generated block (`_COEFFICIENTS_HASH`, kernel functions,
named instances like `seven_point_laplacian`). `stencil.py` should export only
the parameterizable generator `derive_stencil(deriv_order, approx_order, ndim)`.
Workflow: every invoker of stencil functionality — test files, benchmarks,
application code — must invoke code generation directly: call
`derive_stencil(deriv_order, approx_order, ndim)` to get weights, then construct
a Stencil instance. No pre-generated named instances are imported; each callsite
derives what it needs. This distributes code-gen responsibility to the edge,
keeps `stencil.py` clean and focused, and eliminates the need to name and commit
every possible concretization.

**Implement auto-discovery for kernel module generation and testing.**
With `stencil.py` cleaned up and the framework proven, generalize the generator
script and test suite to auto-discover and process all kernel modules in one
pass:

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
Audit every existing operator (`cosmic_foundry/computation/*.py` excluding
`_codegen.py`) and apply the derivation pattern: add `_derive()` that returns
constants and stencil structure, and `generate()` that emits the full block. The
parameterized auto-discovery test will automatically verify each one. Any
operator without a `_derive()` is not yet compliant with architectural basis
claim 5 (every numerical method formally derived).

Start with operators that have hardcoded constants (stencils, reductions) where
the pattern applies immediately. Operators without a formal derivation (e.g.,
field sampling, overlap operations) may require a minimal `_derive()` that
documents why derivation is not applicable.
