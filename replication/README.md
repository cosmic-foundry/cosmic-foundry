# Replication workflow

This directory holds the workflow, specs, plans, and golden-data
harnesses for replicating other astrophysics codes' results to
within numerical roundoff. The workflow is governed by
[ADR-0007](../adr/ADR-0007-replication-workflow.md) and is itself
a **living specification**: update this README as we learn from
each replication target.

## Layout

```text
replication/
  README.md                      # this file (the workflow)
  capabilities/                  # shared engine specs (one per capability)
    CNNNN-<capability>.md
  targets/
    <target-code>/
      plan.md                    # ordered problem list + capability dependencies
      problems/
        PNN-<problem>.md         # one spec per test problem
      golden/
        manifest.yaml            # fixture_id → (target version, generator, hash, default tolerance)
        generators/              # scripts that produce reference outputs from the target
        fixtures/                # the reference outputs themselves (or LFS pointers)
      conftest.py                # exposes load_golden(fixture_id) → array + metadata
```

**Capabilities** are engine features (a PPM solver, a reaction
network, a passive-tracer advector). Each capability has one
canonical spec in `capabilities/` and is referenced by ID
(`CNNNN`) from the problem specs that depend on it.

**Problems** are the test cases we replicate from a target code
(e.g. a KH Galilean-invariance setup, a 1D nuclear detonation).
Each target has a `plan.md` that orders its problems and names
the capabilities each consumes.

A capability's status reflects the union of its dependents:
`Verified` once at least one problem depending on it passes its
verification; `Drifted` if any dependent starts failing.
Cross-target regressions surface without bookkeeping.

## Spec templates

Two kinds of spec. Capability specs describe engine features in
isolation; problem specs describe a target code's test case and
name the capabilities they consume.

### Capability spec

Copy into `replication/capabilities/CNNNN-<capability>.md`.

```markdown
# Capability: <name>

- **ID:** CNNNN
- **Status:** Proposed | Implementing | Verified | Drifted
- **Implemented in:** <engine module path, once it exists>

## Behavior

What the feature does, independently of any target. Inputs,
outputs, units. Algorithmic choices (e.g. PPM vs MUSCL) when the
choice is load-bearing.

## Numerical signature

- Expected convergence order(s).
- Conservation / symmetry invariants and their tolerances.
- Interface contract with other capabilities it couples to.
- **External reference:** (required for physics capabilities before
  status advances past Implementing) — one of:
  - Analytical solution with derivation or citation.
  - Published benchmark value: paper, equation or table number, value.
  - Symbolic derivation: continuous operator → discrete stencil via
    SymPy or equivalent, with the derivation committed alongside.
  See ADR-0007 §Amendments for the definition of "externally grounded."

## Dependents

Problems that consume this capability. Kept in sync with problem
specs' `Capabilities required` fields.

## Open questions

Unresolved choices. Removed once resolved.
```

### Problem spec

Copy into `replication/targets/<target>/problems/PNN-<problem>.md`.

```markdown
# Problem: <name>

- **ID:** PNN (unique within target)
- **Status:** Proposed | Implementing | Verified | Drifted
- **Target code:** <name@version-or-commit>
- **References:** [paper §, source file:line, …]
- **Capabilities required:** [CNNNN, CNNNN, …]

## Setup

Initial conditions, domain, boundary conditions, units. Any
target-specific conventions (sign, coordinate, scaling) that are
load-bearing for reproduction. Known target-side quirks to
replicate rather than "fix".

## Success criterion

What exactly must be reproduced, with tolerances. This is the
numerical claim the PR must verify.

## Verification plan

- Unit fixtures: (input, golden output, tolerance).
- Convergence test: grid sequence and expected slope.
- Target-specific diagnostics (mode amplitude, detonation speed,
  etc.).

## Out of scope

Adjacent problems deliberately deferred.

## Open questions

Things to resolve before or during implementation. Removed once
resolved; surviving entries indicate the spec is still living.
```

The `Status` and `Open questions` fields on both kinds of spec
are how we keep them honest as we learn — entries get updated,
not rewritten. `Drifted` on a capability or problem flags a
previously-verified artifact whose fixtures or invariants have
started failing; opening that status is a signal to investigate
before further work in the area.

## Golden-data harness rules

- **Version-pinned.** `manifest.yaml` records the target code's
  version or commit for every fixture.
- **Reproducible.** Every fixture has a generator script under
  `generators/` that, given the pinned target version, produces
  the same bytes (or a recorded hash if the target is not
  bit-deterministic).
- **Single loader.** Tests load fixtures only via
  `load_golden(fixture_id)` from `conftest.py`. Loading a
  fixture absent from the manifest is a test failure, not a
  warning.
- **Storage.** Small fixtures (<~1 MB) are checked in directly.
  Larger fixtures route through Git LFS, consistent with
  ADR-0006's visual-regression assets policy.
- **Tolerances are explicit.** The manifest carries a default
  tolerance per fixture; tests may tighten but should not
  silently loosen it.
- **Regression detection, not correctness grounding.** Golden
  files catch drift after a correct baseline is established.
  They do not establish correctness on their own: a fixture
  generated from a wrong implementation encodes a wrong answer.
  Every physics capability must also have at least one
  externally grounded test (see ADR-0007 §Amendments and the
  External reference field in the capability spec template).

## Exceptions to the bounded-increment rule

The ≤100-line + ship-with-verification discipline holds in the
common case. The following exceptions exist; each must be named
in the PR description so reviewers know which rule is being
relaxed and why.

1. **Emergent integration-only drift.** A bug that appears only
   after composition (operator-splitting error, accumulated
   roundoff over many timesteps). The spec still exists; the
   verification artifact moves to the integration level. The
   PR may exceed 100 lines if the fix is genuinely
   cross-cutting, and must ship with the integration test that
   pins the new behavior.
2. **Calibration / matching passes.** When the work is "find
   why our trajectory diverges from the target at t=X," the
   loop is exploratory: read intermediate state, diff,
   hypothesize, re-run. Do this on a scratch branch; land the
   *outcome* (a fix or a new spec entry plus its test) as a
   normal small PR.
3. **Reference-code archaeology.** Running the target
   side-by-side to understand undocumented behavior. Output is
   notes and spec updates, not engine code.
4. **Trivial changes.** Typo fixes, dependency bumps,
   formatting. No spec needed; verification is "tests still
   pass."
5. **Pure scaffolding.** The harness itself, ADRs, plan docs,
   and this README. No numerical claim to verify, so PR size
   may exceed 100 lines.

**Invariant across all exceptions:** never both skip the spec
and skip the verification. One of them must exist for the
change to land.

## Living-spec discipline

This README and ADR-0007 will be wrong about something within
the first few replication targets. When a rule turns out to be
unworkable or insufficient:

- Open a PR amending this README and link it from the
  replication target whose experience motivated the change.
- For load-bearing changes (e.g. a new exception class, a
  redefinition of "verification artifact"), open a follow-on
  ADR superseding the relevant section of ADR-0007 rather than
  silently editing this file.
