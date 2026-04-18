# Meta-Level Roadmap

This plane is the roadmap for reproducibility, verification, validation,
provenance, and evidence. It answers **how Cosmic Foundry proves and
regenerates object-level claims**.

The companion [object-level roadmap](../object-level/README.md) answers
what platform and simulation capabilities the codebase is building.

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
| M0 | [ADR-0005](../../adr/meta-level/ADR-0005-branch-pr-attribution-discipline.md), [`AI.md`](../../AI.md) | Branch, PR, commit-size, history, and attribution discipline. | Active |
| M1 | [ADR-0007](../../adr/meta-level/ADR-0007-replication-workflow.md), [`replication/`](../../replication/README.md) | Bounded-increment verification, capability specs, golden-data harness, formulas register, externally grounded tests. | Active |
| M2 | [ADR-0013](../../adr/meta-level/ADR-0013-derivation-first-lane.md), `derivations/` | Lane A/B/C provenance discipline and derivation documents for physics capabilities. | Active; first derivation pending |
| M3 | This file | Capability intent documentation: for each existing platform capability, a clear spec stating what it computes and how an independent actor would verify it. The capsule tooling (M3b) is deferred until claims are documented clearly enough to be worth collecting. | Current focus |
| M3b | [reproducibility-meta-generator.md](reproducibility-meta-generator.md), [ADR-0015](../../adr/meta-level/ADR-0015-reproducibility-meta-generator.md) | Reproducibility capsules, collect/dry-run/render/compare, recursive approximate idempotence. | Planned; depends on M3 |
| M4 | [epoch-03-platform-services.md](../object-level/epoch-03-platform-services.md) | Validation manifests, provenance sidecars, comparison-result schema, simulation-specification format. | Planned for Epoch 3 |
| M5 | Application-repo capsule integration | Application capability capsules, validation products, evidence idempotence, multi-repository regeneration. | Future |

## Current Focus: M3 Capability Intent Documentation

The capsule tooling (M3b) is only valuable if the claims it collects are
clearly stated. The current focus is therefore a step earlier: for each
existing platform capability, write a spec that states what it computes and
how an independent actor — without access to the authors or the git history
— could verify that the implementation is correct.

The test: could someone read the spec, implement the capability themselves,
run the stated verification, and know whether they got it right? If not, the
spec is incomplete.

Once existing capabilities meet that bar, M3b (the capsule tooling) has
something real to collect and dry-run against.

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
