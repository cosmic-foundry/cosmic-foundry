# Replication workflow

This directory holds the workflow, specs, plans, and golden-data
harnesses for replicating other astrophysics codes' results to
within numerical roundoff. The workflow is governed by
[ADR-0007](../adr/meta-level/ADR-0007-replication-workflow.md) and is itself
a **living specification**: update this README as we learn from
each replication target.

## Layout

```text
replication/
  README.md                      # this file (the workflow)
  formulas.md                    # flat register of individual physics formulas
  targets/
    <target-code>/
      plan.md                    # ordered problem list + maps required
      problems/
        PNN-<problem>.md         # one spec per test problem
      golden/
        manifest.yaml            # fixture_id → (target version, generator, hash, default tolerance)
        generators/              # scripts that produce reference outputs from the target
        fixtures/                # the reference outputs themselves (or LFS pointers)
      conftest.py                # exposes load_golden(fixture_id) → array + metadata
```

**`formulas.md`** is a flat cross-reference of individual physics
formulas: one entry per equation, stencil, rate, or fit coefficient
whose implementation could silently encode a wrong answer. It is
finer-grained than a single `Map:` block — one map may generate
several formula entries. See `formulas.md` for the entry criteria
and the current register.

**Maps** are the physics operators the engine implements. What a map
computes is described by its `Map:` block in the implementing module
(domain, codomain, operator, Θ, p). Before implementing a map, write
its `Map:` block — that is the spec. For Lane B/C maps (ADR-0013),
also commit a derivation document under `derivations/` before code.

**Problems** are the test cases we replicate from a target code
(e.g. a KH Galilean-invariance setup, a 1D nuclear detonation).
Each target has a `plan.md` that orders its problems and names
the maps each requires.

## Spec templates

### Map spec (physics operator)

The spec for a physics map lives in the implementing module as a
`Map:` block (ADR-0016). Before writing code, write the `Map:` block:

```python
class MyPhysicsMap:
    """Brief description.

    Map:
        domain   — description of inputs
        codomain — description of output
        operator — what the operator does

    Θ = {h, …} — approximation parameters.
    p = N       — convergence order.

    External reference: <paper, §, equation number>.
    """
```

For Lane B/C maps (ADR-0013), also commit a derivation document
under `derivations/` before opening the implementation PR.

### Problem spec

Copy into `replication/targets/<target>/problems/PNN-<problem>.md`.

```markdown
# Problem: <name>

- **ID:** PNN (unique within target)
- **Status:** Proposed | Implementing | Verified | Drifted
- **Target code:** <name@version-or-commit>
- **References:** [paper §, source file:line, …]
- **Maps required:** [module.ClassName, …]

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
