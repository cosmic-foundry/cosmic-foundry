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
  README.md                    # this file (the workflow)
  <target-code>/
    plan.md                    # ordered checklist of capability specs
    specs/
      NNNN-<capability>.md     # one spec per capability
    golden/
      manifest.yaml            # fixture_id → (target version, generator, hash, default tolerance)
      generators/              # scripts that produce reference outputs from the target
      fixtures/                # the reference outputs themselves (or LFS pointers)
    conftest.py                # exposes load_golden(fixture_id) → array + metadata
```

One directory per target code. Specs are numbered for stable
ordering; the plan provides the human-readable sequence.

## Spec doc template

Copy this skeleton into `replication/<target>/specs/NNNN-<capability>.md`.

```markdown
# Capability: <name>

- **Status:** Proposed | Implementing | Verified | Drifted
- **Target code:** <name@version-or-commit>
- **References:** [paper §, source file:line, …]

## Behavior

What the target does, in equations and prose. Inputs, outputs,
units. Be explicit about coordinate conventions, sign
conventions, and any quirks that look like bugs but are
load-bearing.

## Numerical signature

- Expected convergence order(s) and the test that proves it.
- Conservation / symmetry invariants and their tolerances.
- Known target-side failure modes (so we *replicate*, not "fix").

## Verification plan

- Unit fixtures: list of (input, golden output, tolerance).
- Convergence test: grid sequence and expected slope.
- Integration anchor: which scenario this capability appears in.

## Out of scope

Adjacent capabilities deliberately deferred to other specs.

## Open questions

Things to resolve before or during implementation. Removed once
resolved; surviving entries indicate the spec is still living.
```

The `Status` and `Open questions` fields are how we keep specs
honest as we learn — entries get updated, not rewritten.
`Drifted` flags a previously-verified capability whose fixtures
or invariants have started failing; opening that status is a
signal to investigate before further work in the area.

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
