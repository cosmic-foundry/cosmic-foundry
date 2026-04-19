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

**Generalize the derivation infrastructure into a reusable framework.**
The sentinel splicing logic, hash computation, and `generate()` contract are
currently hand-rolled inside `laplacian.py` and `scripts/generate_kernels.py`.
Before rolling out to additional operators, extract these into shared utilities:
a `_make_hash(constants)` helper and a `generate_constants_block(constants)`
formatter in `cosmic_foundry/computation/_codegen.py`; sentinel strings as
module-level constants shared between the generator script and the drift-check
test; and `scripts/generate_kernels.py` generalized to discover and splice any
module that exposes a `generate()` function. Each new kernel module then only
needs to implement `_derive()` and call the shared utilities — no boilerplate
to copy.

**Scale derivation pattern to the full codebase.**
`cosmic_foundry/computation/laplacian.py` establishes the proof-of-concept:
derivation (`_derive`), hash-verified generated constants block, and production
kernel in a single file; `scripts/generate_kernels.py` splices fresh constants
via `BEGIN GENERATED` / `END GENERATED` sentinels. The next step is to audit
every existing operator and apply the same pattern: add `_derive()` and
`generate()`, generate the constants block, and add a drift-check entry in
`tests/test_generated_kernels.py`. Any operator without a `_derive()` is not
yet compliant with architectural basis claim 5.
