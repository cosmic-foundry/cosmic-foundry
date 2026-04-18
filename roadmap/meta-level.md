# Meta-Level Roadmap

This plane is the roadmap for reproducibility, verification, validation,
provenance, and evidence. It answers **how Cosmic Foundry proves and
regenerates object-level claims**.

The companion [object-level roadmap](object-level.md) answers what
platform and simulation capabilities the codebase is building.

## Framing

**Goal.** Build an evidence system that can describe, verify, validate,
and regenerate a physics engine state across the platform and application
repository family.

**Principle - executable evidence before scale.** Object-level claims
should be backed by executable meta-level evidence as early as possible:
specs, formulas, derivations, golden fixtures, validation manifests,
capsules, dry-run checks, and eventually regenerated evidence bundles.

**Principle - track selection per PR.** Each PR declares which track it
advances: object-level, meta-level, or both. The tracks can proceed in
parallel, but cross-track dependencies must be explicit.

**Principle - object-level claims stay out of the meta plane.** The
meta-level roadmap defines evidence machinery and acceptance workflow. It
does not decide scientific scope, numerical method choice, or platform
capability ordering except by recording explicit evidence dependencies.

## Stages

| ID | File / Authority | Scope | Status |
|----|------------------|-------|--------|
| M0 | [ADR-0005](../adr/ADR-0005-branch-pr-attribution-discipline.md), [`AI.md`](../AI.md) | Branch, PR, commit-size, history, and attribution discipline. | Active |
| M1 | [ADR-0007](../adr/ADR-0007-replication-workflow.md), [`replication/`](../replication/README.md) | Bounded-increment verification, capability specs, golden-data harness, formulas register, externally grounded tests. | Active |
| M2 | [ADR-0013](../adr/ADR-0013-derivation-first-lane.md), `derivations/` | Lane A/B/C provenance discipline and derivation documents for physics capabilities. | Active; first derivation pending |
| M3 | [reproducibility-meta-generator.md](reproducibility-meta-generator.md), [ADR-0015](../adr/ADR-0015-reproducibility-meta-generator.md) | Reproducibility capsules, collect/dry-run/render/compare, recursive approximate idempotence. | Planned meta-level focus |
| M4 | [epoch-03-platform-services.md](epoch-03-platform-services.md) | Validation manifests, provenance sidecars, comparison-result schema, simulation-specification format. | Planned for Epoch 3 |
| M5 | Application-repo capsule integration | Application capability capsules, validation products, evidence idempotence, multi-repository regeneration. | Future |

## Current Focus: M3 Platform Convergence

The current planned meta-level work is the M3 platform convergence slice:

```text
collect current platform state
-> dry-run capsule references
-> render independent-actor instructions
-> collect again from the same or regenerated state
-> compare normalized capsules for structural idempotence
```

The detailed implementation plan and exit criteria live in
[reproducibility-meta-generator.md](reproducibility-meta-generator.md).

## Cross-Plane Interfaces

Meta-level work reads object-level state through explicit contracts:

- ADR indexes and architecture planes;
- `STATUS.md` current-position metadata;
- object-level roadmap files and epoch implementation plans;
- replication capability specs and target plans;
- formula register entries;
- derivation documents;
- manifest schemas and provenance sidecars;
- test selectors, convergence expectations, and validation thresholds;
- source maps and environment recipes.

Meta-level work writes evidence artifacts:

- reproducibility capsules;
- dry-run diagnostics;
- rendered independent-actor instructions;
- execution transcripts;
- evidence indexes;
- normalized capsule comparison reports.
