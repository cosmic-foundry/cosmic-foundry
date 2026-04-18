# Reproducibility Meta-Generator Roadmap

> Meta-level verification implementation plan. See
> [ADR-0015](../adr/ADR-0015-reproducibility-meta-generator.md).

## Purpose

This is the first focused implementation plan in the meta-level roadmap:
the track concerned with reproducibility, verification, validation,
provenance, and evidence rather than object-level engine capabilities.

Build enough of the reproducibility meta-generator to exercise it against
the platform state that exists today. The goal is not to finish the full
long-term capsule executor. The goal is to make the regeneration workflow
executable early enough that it can discover missing contracts in the
platform's ADRs, roadmap, replication specs, schemas, environment recipe,
and tests.

This meta-level plan should converge on a working platform-only capsule
workflow:

```text
collect current platform state
-> dry-run capsule references
-> render independent-actor instructions
-> collect again from the same or regenerated state
-> compare normalized capsules for structural idempotence
```

## Scope

In scope for the first implementation slice:

- platform-only capsule schema and validation;
- CLI surface for `collect`, `dry-run`, `render`, and `compare`;
- source-map collection from git metadata;
- architecture-basis collection from ADR index, `STATUS.md`, and roadmap
  files;
- environment recipe collection from committed environment scripts and
  packaging metadata;
- inventory of platform schemas and replication specs;
- dry-run checks that referenced files and declared commands exist;
- normalized structural comparison of two capsules; and
- tests proving that the current repository can produce a structurally
  stable capsule.

Out of scope for the first implementation slice:

- application repository collection;
- domain validation products;
- expensive physics execution;
- evidence idempotence;
- network data fetches;
- container or VM image production; and
- multi-repository regeneration beyond recording source-map entries.

## Implementation Plan

The plan is intentionally short. Each item should be a small PR unless
implementation experience shows two adjacent items are only useful
together.

1. **ADR and roadmap seed** — define the meta-generator architecture,
   recursive approximate idempotence, and this implementation plan.
   *Depends on: none.*

2. **Capsule schema and model** — add a versioned platform capsule schema
   plus Python model helpers for identity, source map, architecture
   basis, environment recipe, capability inventory, verification plan,
   validation plan placeholder, execution transcript placeholder, and
   evidence index placeholder. Include normalization rules for volatile
   fields, but do not collect repository data yet.
   *Depends on: #1.*

3. **CLI skeleton and renderer** — add `cosmic-foundry capsule` commands
   for `collect`, `dry-run`, `render`, and `compare`. At this stage
   `collect --target platform` may emit a minimal synthetic capsule so
   command shape, schema validation, and human-readable instructions are
   reviewable before the collector becomes smart.
   *Depends on: #2.*

4. **Platform collector** — implement real platform collection from the
   current repository: git source map, ADR index, `STATUS.md`, roadmap
   files, environment scripts, packaging metadata, platform schemas,
   replication specs, and baseline verification commands. The collector
   records missing optional sections as explicit gaps rather than silently
   omitting them.
   *Depends on: #3.*

5. **Dry-run validators** — validate collected references without running
   expensive work: linked files exist, schema files parse, ADR index links
   resolve, replication specs are discoverable, declared verification
   commands are present, and capsule-required sections are populated or
   explicitly skipped with a reason.
   *Depends on: #4.*

6. **Structural comparison** — implement normalized capsule comparison.
   Normalize generated timestamps, local absolute paths, temporary
   directories, wall-clock durations, host/user details, and unordered
   collections whose order has no semantic meaning. Report unexplained
   nonvolatile divergence as a failing comparison.
   *Depends on: #5.*

7. **Platform convergence test** — add a test or script that collects a
   platform capsule, dry-runs it, renders instructions, collects again
   from the same repository state, and proves structural idempotence after
   normalization. If a regenerated checkout can be created cheaply from
   the source map, add that as an optional integration test; otherwise
   record it as the next expansion point.
   *Depends on: #6.*

8. **Track status update** — run the platform convergence workflow and
   update `STATUS.md` with the result. Record whether the next selected
   PR advances the meta-level track, the object-level track, or both.
   *Depends on: #7.*

## Exit Criteria

The M3 platform convergence slice is complete when:

- `cosmic-foundry capsule collect --target platform` emits a valid
  capsule for the current repository;
- `cosmic-foundry capsule dry-run <capsule>` fails on missing required
  metadata and reports explicit skips for unsupported sections;
- `cosmic-foundry capsule render <capsule>` emits usable independent-
  actor instructions;
- `cosmic-foundry capsule compare <a> <b> --mode structural` normalizes
  volatile fields and fails on unexplained nonvolatile divergence;
- CI or the local test suite exercises the platform capsule convergence
  workflow; and
- the resulting gaps are fixed in authoritative artifacts or listed with
  removal conditions.

## Relationship To The Object-Level Roadmap

This roadmap does not replace the object-level Epoch 2 roadmap. It is a
parallel meta-level plan. The task-graph driver will become a central
orchestration point for later physics verification, so the meta-generator
should eventually be able to describe that platform state and its
verification plan, but object-level and meta-level PRs remain separate
track choices unless a PR records a specific cross-track dependency.

The current object-level roadmap remains:

1. task-graph driver - single-rank;
2. multi-rank halo fill;
3. AMR hierarchy and I/O items already listed in
   [epoch-02-mesh.md](epoch-02-mesh.md).
