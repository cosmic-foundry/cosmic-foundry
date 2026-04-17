## ADR-0007 — Replication workflow: bounded-increment, verification-first

- **Status:** Accepted
- **Date:** 2026-04-14

## Context

A common engine task is replicating another astrophysics
code's results to within numerical roundoff. Doing this
reliably requires both careful planning of *what* the target
code does and a verification structure that catches numerical
drift before it accumulates across capabilities or epochs.

The 100-line commit guideline (PR #8) already encodes part of
the answer: small, reviewable units. The motivation behind it
is broader, however — long uninterrupted AI sessions and
long uninterrupted human edits both degrade in accuracy
without frequent grounding against verifiable evidence. The
guideline as written says *how big* a change can be but does
not say *what evidence* must accompany it, and so leaves
numerical correctness to integration-time debugging.

This ADR generalizes the bounded-increment idea into a
workflow that also pins what each increment must prove.

## Decision

Adopt a **bounded-increment, verification-first** workflow
for all engine work that makes a numerical claim. Concretely:

- **Spec before code.** Engine capabilities (a solver, a
  reaction network, a tracer advector) get canonical specs in
  `replication/capabilities/`; test problems from target codes
  (a KH setup, a 1D detonation) get specs under
  `replication/targets/<target>/problems/` that reference their
  capabilities by ID. Capability specs describe the engine
  feature in isolation: algorithm, inputs and units, expected
  numerical signature (convergence order, conservation
  invariants, interface contract). Problem specs describe the
  target-code setup, references (paper section, source
  file:line), known target-side failure modes to replicate
  rather than "fix", and the verification plan (fixtures,
  convergence tests, target-specific diagnostics).
- **Golden-data harness.** Reference outputs are produced by
  scripted, version-pinned generators, recorded in a
  manifest with content hashes, and consumed via a single
  fixture-loading API. Fixtures are never hand-edited; tests
  that load a fixture absent from the manifest fail.
- **Verification-driven PRs.** Every PR landing a numerical
  claim ships with a test that compares against golden data
  within a stated tolerance, plus the relevant convergence
  or invariant test where the spec calls for one. The
  ≤100-line size guidance still applies; it may stretch when
  a fix is genuinely cross-cutting, but the verification
  artifact is non-negotiable.
- **Numerical guardrails beyond point checks.** Convergence-
  order tests, conservation/symmetry invariants, and
  regression sentinels run alongside fixture comparisons.
  Point checks alone do not catch drift.
- **A living plan per replication target.** A checklist doc
  (`replication/targets/<target>/plan.md`) orders the target's
  problems and names the capabilities each consumes; spec
  status fields (Proposed → Implementing → Verified →
  Drifted) update as work progresses. A capability's status
  reflects the union of its dependents — Verified once any
  dependent passes, Drifted if any dependent starts failing —
  so cross-target regressions surface without bookkeeping.

Named exceptions (emergent integration-only drift,
calibration passes, reference-code archaeology, trivial
changes, pure scaffolding) are enumerated in
`replication/README.md`. Invariant across all exceptions:
**never both skip the spec and skip the verification.** One
of them must exist for the change to land.

The workflow itself lives in `replication/` and is a living
specification — `replication/README.md` is updated as we
accumulate experience with replication targets, and ADRs
amending this one are expected.

## Consequences

**Positive.** Numerical drift is caught at the unit level
where it is cheap to diagnose, instead of at integration
time where causes and effects are entangled across
capabilities. PRs remain small and reviewable. Replication
tasks decompose into trackable units that survive across
sessions and across agents. AI-agent runs are continually
grounded against fixtures rather than running blind for
long stretches.

**Negative.** Significant upfront investment per replication
target before the first line of "real" engine code: spec
authoring, generator scripts, fixture storage, manifest
plumbing. Calibration and archaeology work need explicit
"exception" framing in PRs rather than open-ended
exploration on main.

**Neutral.** Introduces a new top-level directory
`replication/`. Adds a class of test (convergence and
invariant) on top of standard unit tests. Large reference
fixtures will route through Git LFS as already established
by ADR-0006 for visual-regression assets.

## Alternatives considered

- **Status quo (size guideline alone).** Keeps PRs
  reviewable but leaves verification ad hoc, so numerical
  drift continues to surface at integration time.
- **Spec-only, no fixture harness.** Captures intent but
  provides no automated grounding signal between PRs;
  regressions are caught only when someone re-runs the
  target code by hand.
- **Fixture-only, no spec docs.** Catches regressions but
  loses the "what are we replicating and why" context,
  making test failures hard to interpret months later and
  making it easy to mistake target-side quirks for bugs.
- **Full per-capability TDD without size discipline.**
  Strong on verification but encourages large,
  hard-to-review PRs and reintroduces the long-uninterrupted-
  edit failure mode the size guideline was written to
  prevent.

## Amendments

- **2026-04-16 — External-grounding requirement for physics
  capabilities.** The original decision required golden-data
  comparison plus convergence or invariant tests "where the
  spec calls for one." That language left a systematic gap:
  when an AI agent writes both the implementation and the
  fixture generator, a golden file encodes whatever the
  implementation produces — correct or not. Golden files
  reliably detect *regression* only after correctness has
  been established by some independent means. This amendment
  tightens the verification requirement for physics
  capabilities specifically.

  **Definitions for this amendment.**

  A *physics capability* is any engine module that
  implements a physical equation, numerical scheme, or
  derived quantity — gradient, divergence, Riemann solver,
  EOS, reaction network, etc. Infrastructure capabilities
  (dispatch, field placement, I/O, mesh topology) are not
  physics capabilities; their correctness is grounded in
  structural and algebraic properties rather than in
  physical equations.

  An *externally grounded test* is a test whose expected
  answer comes from a source the implementation cannot have
  influenced: an analytical solution (with the derivation
  stated or cited), a published benchmark number (with paper
  and equation or table reference), or a result derived
  symbolically from the continuous operator (e.g. via SymPy
  applied to the differential operator before discretization).
  A golden HDF5 file generated by the engine's own code is
  not externally grounded; it is a regression sentinel only.

  **Additional requirements for physics capabilities.**

  1. *Convergence-rate testing is non-optional when the
     method has a stated design order.* The capability
     spec's Numerical Signature section must name the
     expected convergence order. The corresponding test
     must measure the actual exponent over a resolution
     sequence and assert it lies within an acceptable band
     of the expected value. A single-resolution closeness
     check does not satisfy this requirement.

  2. *At least one test per physics capability must be
     externally grounded.* A capability whose only tests
     compare against engine-generated golden data is
     insufficiently verified; its status may not advance
     past Implementing in the capability spec.

  3. *The external grounding source is recorded in the
     capability spec.* The Numerical Signature section
     gains a mandatory **External reference** field (see
     updated template in `replication/README.md`) that
     names the source. Reviewers use this field to confirm
     independently that the expected answer is correct
     before the test value is committed.

  **Scope.** These requirements apply to Epoch 2 and later.
  Epoch 1's infrastructure capabilities are explicitly
  exempt. Note that Epoch 1's Laplacian test already
  satisfies the intent: the expected value of 6.0 is
  derived analytically from ∇²(x²+y²+z²) = 6, not from a
  golden file; this is the model for future physics tests.

- **2026-04-17 — Promoted Proposed → Accepted.** The workflow
  has been in force since adoption: Epoch 1 landed under it
  (infrastructure-exempt per the 2026-04-16 amendment), the
  formula register (`replication/formulas.md`) and verification
  helpers (`tests/utils/convergence.py`, `tests/utils/stencils.py`)
  were built to serve it, and the two Epoch 2 design-prerequisite
  ADRs (0011 HaloFillFence, 0012 DiagnosticReducer) both cite it
  as the framing for their verification obligations. Surfaced as
  a status correction by the Epoch 1 → Epoch 2 ADR-family review
  (2026-04-17); no change to the decision itself.
