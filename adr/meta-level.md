# Meta-Level Architecture

This plane records architecture for reproducibility, verification,
validation, provenance, and review discipline. It answers **how Cosmic
Foundry makes object-level claims auditable and regenerable**.

Object-level architecture lives separately in
[object-level.md](object-level.md) and answers what platform and
simulation capabilities the project is building.

## Process, Verification, And Reproducibility

How work is organized and how object-level claims become auditable.

- [**ADR-0005**](ADR-0005-branch-pr-attribution-discipline.md) —
  Branch, PR, commit-size, history, and attribution discipline for human
  and AI-agent contributors. Authoritative source; `AI.md` is an
  informal quick-reference kept aligned with this ADR.
- [**ADR-0007**](ADR-0007-replication-workflow.md) — Replication
  workflow: bounded-increment, verification-first. Every PR with a
  numerical claim ships with golden-data verification; spec docs and a
  per-target plan live under `replication/`.
- [**ADR-0008**](ADR-0008-numerical-transcription-discipline.md)
  *(stub)* — Numerical-transcription discipline for files like
  `aprox_rates.H` where the diff-size guideline is ill-fitted and
  ADR-0007's verification-first discipline is the real defect defense.
- [**ADR-0013**](ADR-0013-derivation-first-lane.md) — Derivation-first
  lane for physics capabilities. Three named paths: Lane A
  (port-and-verify), Lane B (clean-room from paper), and Lane C
  (first-principles origination). Lanes B and C require derivation
  documents with executable SymPy checks on load-bearing algebraic steps.
- [**ADR-0015**](ADR-0015-reproducibility-meta-generator.md) —
  Reproducibility meta-generator: platform-owned machinery that emits and
  optionally executes versioned reproducibility capsules. Recursive
  capsule comparison targets approximate idempotence after declared
  volatile fields are normalized.

## Bridge Points To The Object-Level Plane

Meta-level architecture constrains object-level work through explicit
evidence contracts:

- object-level numerical capabilities use ADR-0007 capability specs,
  formulas, fixtures, convergence tests, and externally grounded tests;
- physics-capability PRs choose an ADR-0013 lane and, when required,
  carry derivation documents;
- platform manifest infrastructure from ADR-0014 supplies the
  object-level machinery that meta-level validation and provenance
  contracts use; and
- ADR-0015 capsules collect object-level source maps, ADR basis,
  capability manifests, verification plans, validation plans, execution
  transcripts, and evidence indexes.
