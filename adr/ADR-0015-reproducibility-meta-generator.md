# ADR-0015 - Reproducibility meta-generator

## Context

Cosmic Foundry already has several reproducibility artifacts, but they
are distributed across the repository family:

- ADRs record architectural commitments.
- `STATUS.md` and roadmap files record project position and sequencing.
- `replication/` records capability specs, target plans, formulas, and
  verification expectations.
- `derivations/` is reserved by ADR-0013 for capability-level derivations
  with executable symbolic checks.
- `cosmic_foundry.manifests` owns domain-neutral manifest, provenance,
  validation, and simulation-specification infrastructure.

Those artifacts reduce drift while a developer is working inside the
repository, but they do not yet answer a higher-level reproducibility
question:

> What compact instruction set would let an independent actor regenerate
> a physics engine at a particular point in time, execute as much of the
> regeneration and verification workflow as possible, and discover which
> architectural, numerical, environmental, or provenance claims are
> incomplete?

This is not only a publishing problem. The project needs this capability
during development so the workflow can fail early when a capability spec
misses a required formula, a derivation lacks an executable check, a
validation product has no provenance sidecar, an environment cannot be
recreated, or an application repository depends on platform behavior that
is not part of a declared contract.

The platform/application split in ADR-0014 also constrains ownership:
the platform can own the generic regeneration machinery and schemas, but
application repositories own domain physics, domain validation products,
and domain-specific pass/fail criteria.

## Decision

Adopt a **reproducibility meta-generator** as platform infrastructure.
The meta-generator emits and optionally executes a versioned
**reproducibility capsule**: a compact, machine-readable instruction set
plus human-readable notes that describe how to regenerate and verify a
physics engine or capability set at a specific point in time.

### Reproducibility capsule

A capsule is an evidence-oriented contract, not a source archive. It
records enough metadata for an independent actor to fetch the required
repositories, reconstruct the declared environment, execute the declared
verification and validation commands, and inspect the resulting evidence.

The first capsule version contains these sections:

1. **Identity.** Capsule schema version, generated timestamp, generator
   version, target name, and target type (`platform`, `application`,
   `cross-scale-workflow`, or `capability-set`).
2. **Source map.** Repository URLs, commits, expected remotes, optional
   release tags, and local path assumptions expressed only as
   repository-relative paths.
3. **Architecture basis.** ADR index entries that govern the target,
   roadmap or status documents that define scope, and any target-specific
   design notes.
4. **Environment recipe.** Environment setup commands, lock files when
   available, required optional extras, hardware assumptions, and known
   nondeterminism or accelerator constraints.
5. **Capability manifest.** Capability IDs, provenance lane (`A`, `B`,
   or `C` where ADR-0013 applies), linked formulas, linked derivations,
   externally grounded tests, and explicit exclusions.
6. **Verification plan.** Commands, test selectors, expected artifacts,
   tolerances, convergence-order expectations, conservation checks,
   regression fixtures, and the rule that distinguishes an externally
   grounded test from an engine-generated regression sentinel.
7. **Validation plan.** Validation products, observational provenance,
   comparison-result schema version, scientific question, acceptance
   threshold, and invalid regimes. Domain-specific content is supplied by
   application repositories; the platform supplies the schema.
8. **Execution transcript.** Optional run records produced when the
   capsule is executed: command, exit status, duration, output artifact
   hashes, diagnostics, and failure summaries.
9. **Evidence index.** Paths and hashes for test reports, diagnostic
   files, plots, comparison tables, provenance sidecars, and generated
   bibliography.

The capsule has two rendered forms:

- a JSON or YAML document used by automation; and
- an `instructions.md` rendering for independent humans.

### Generator and executor modes

The platform provides a single conceptual tool with four modes:

- **Collect.** Walk the declared repositories and assemble the capsule
  from ADRs, replication specs, derivation metadata, manifests, schemas,
  environment files, and git metadata. Collection may fail if required
  metadata is missing.
- **Dry-run.** Validate that referenced files, commands, schema versions,
  formulas, derivations, fixtures, provenance sidecars, and test selectors
  exist without executing expensive physics jobs. Dry-run is the default
  development habit because it is cheap enough to run often.
- **Execute.** Run the declared verification and validation commands,
  write an execution transcript, hash produced artifacts, and update the
  evidence index. Execution may be partial when hardware, data-access, or
  application-repository dependencies are unavailable; partial execution
  must report skipped requirements explicitly.
- **Compare.** Normalize two capsules and report whether they are
  structurally or evidentially equivalent. Compare is the mechanism that
  tests whether a regenerated engine state can reproduce the same
  reproducibility contract.

The first implementation target should be platform-only dry-run support.
It should be able to generate a capsule for the current repository and
detect obvious metadata holes before any application repository is needed.

### Recursive closure and approximate idempotence

The meta-generator should be recursively runnable. A capsule generated
from repository state `X` should contain enough information for an
independent actor to regenerate an equivalent engine state `X'`. Running
the meta-generator again on `X'` should produce a second capsule that is
equivalent to the first after normalizing declared volatile fields.

The intended loop is:

```text
repository state X
-> collect capsule A
-> dry-run or execute capsule A
-> regenerate repository-equivalent state X'
-> collect capsule B from X'
-> compare capsule A and capsule B
```

This property is **approximate idempotence**, not byte-for-byte identity.
The comparison normalizes fields that are expected to vary between runs:

- generated timestamps;
- local absolute paths and temporary directories;
- machine hostname and user-specific environment details;
- wall-clock durations;
- nondeterministic ordering that has no semantic meaning;
- hardware performance metadata; and
- floating-point diagnostics within declared tolerances.

All nonvolatile claims should converge: source commits, architecture
basis, capability set, formulas, derivations, verification commands,
validation products, acceptance thresholds, environment recipe identity,
pass/fail/skip classifications, and deterministic artifact hashes.

The platform should support two comparison levels:

- **Structural idempotence.** The same capsule sections, schemas,
  references, commands, capability links, exclusions, and thresholds are
  present after normalization.
- **Evidence idempotence.** Re-executed checks produce equivalent
  pass/fail/skip classifications and artifact hashes, except for fields
  declared volatile or tolerance-bound.

Structural idempotence is the first implementation target. Evidence
idempotence matures as verification and validation execution mature.
Divergence is a defect unless it is explained by an explicit source,
environment, data-version, or hardware change recorded in the comparison
report.

### Ownership boundaries

Cosmic Foundry owns:

- capsule schemas and versioning;
- source-map, environment, provenance, formula, derivation, and evidence
  section formats;
- generic collection, dry-run, execution-transcript, hashing, and
  rendering machinery;
- capsule normalization and comparison rules for approximate
  idempotence;
- platform validation of capsule structure and references; and
- base commands for platform verification.

Application repositories own:

- domain physics capability content;
- domain-specific derivations and formula entries;
- domain validation products and observational data manifests;
- domain-specific simulation specifications;
- domain-specific acceptance thresholds; and
- any expensive or hardware-specific verification commands beyond the
  platform baseline.

Cross-scale workflow repositories own composition of multiple application
repositories. The platform meta-generator can read their capsule inputs,
but it does not decide their scientific scope.

### Failure semantics

The meta-generator is valuable only if failures are visible. A capsule
run distinguishes:

- **Incomplete metadata.** A required section, linked file, formula,
  derivation, manifest, provenance sidecar, or test selector is missing.
- **Unreproducible environment.** The declared environment cannot be
  recreated or required hardware is unavailable.
- **Verification failure.** A command ran and failed its numerical,
  convergence, invariant, or regression check.
- **Validation failure.** A comparison against an observational or
  published benchmark product failed its declared acceptance threshold.
- **Skipped requirement.** A declared check could not be run for a stated
  reason. Skips are not passes and must appear in the evidence summary.
- **Idempotence divergence.** Two normalized capsules differ in a
  nonvolatile claim without a declared source, environment, data-version,
  or hardware change.

### Iteration rule

The meta-generator begins deliberately incomplete. Each time a capsule
dry-run or execution discovers a missing contract, the project should
prefer one of two fixes:

1. add the missing metadata to the existing authoritative artifact; or
2. amend the capsule schema if the missing concept is genuinely new.

Ad hoc suppression lists are allowed only for named temporary gaps with a
removal condition.

## Consequences

- **Positive:** Reproducibility becomes executable during development.
  Missing specs, stale formulas, absent provenance, environment drift, and
  invalid test selectors become routine failures instead of late review
  surprises.
- **Positive:** The platform/application boundary remains intact. The
  platform owns machinery and schemas; application repositories own
  scientific content and domain acceptance criteria.
- **Positive:** The design supports independent actors without requiring
  them to trust the original authoring session. The capsule points to
  sources, commands, evidence, hashes, and known skips.
- **Positive:** The dry-run mode gives a low-cost habit that can run before
  expensive validation or multi-repository execution exists.
- **Positive:** Recursive comparison keeps the meta-generator from
  becoming a report generator. If regenerated states produce different
  nonvolatile capsules, the workflow has found a reproducibility defect.
- **Negative:** This adds another contract surface. Capsule schemas,
  renderers, and collectors must evolve carefully or they will become
  stale metadata rather than verification machinery.
- **Negative:** Full execution can be operationally expensive. Some
  validation products need network access, large data, accelerator
  hardware, or application repositories that are unavailable in a given
  development session.
- **Neutral:** The first useful implementation is allowed to be incomplete:
  platform-only collection and dry-run are enough to reveal missing
  contracts and drive iteration.

## Alternatives considered

**Manual reproducibility guide only.** Write a human-maintained document
describing how to rebuild and test the engine. This is the lowest
complexity option and remains useful as rendered output, but by itself it
does not fail when a formula, derivation, fixture, provenance sidecar, or
test selector goes missing. It reduces reviewer load only while the guide
is fresh.

**CI-only verification.** Treat the existing test suite and pre-commit
configuration as the reproducibility system. This catches many local code
regressions, but it does not describe how an independent actor should
reconstruct a multi-repository engine, does not classify validation
products, and does not preserve an evidence bundle tied to source and
environment metadata.

**Full environment snapshot or container image.** Publish a complete
container, virtual machine, or source archive. This can improve operational
repeatability, but it hides the provenance graph inside a large artifact
and does not by itself answer whether the physics claims are externally
grounded, derivation-backed, or validated against the declared products.
Capsules may point to containers later; containers are not the contract.

**Application-owned generators.** Let each application repository build
its own regeneration workflow independently. This lowers platform scope
but duplicates schemas, provenance handling, evidence summaries, and
failure semantics across domains. It also makes cross-scale workflows
harder because independent capsule formats would need another
normalization layer.

**Immediate full executor.** Build the executor before the capsule schema
and dry-run checks are stable. Rejected because it increases operational
complexity before the project knows which metadata contracts are missing.
Dry-run collection gives faster feedback and keeps the first iteration
reviewable.

**No recursive comparison.** Generate capsules and evidence reports but
do not require the generator to run on regenerated states. This is simpler
and still improves documentation, but it cannot detect when the
regeneration instructions fail to reconstruct the same contract. Rejected
because approximate idempotence is the check that makes the
meta-generator itself verifiable.

## Architecture stress-review note

### 1. Problem Boundary

The meta-generator solves the problem of converting distributed project
claims into an executable reproducibility contract. It reduces reviewer
load, operational ambiguity, and correctness risk by making missing
metadata and failed checks explicit. A **capsule** is the generated
instruction and evidence contract. **Collect** means assembling metadata
from authoritative artifacts. **Dry-run** means validating references
without expensive execution. **Execute** means running declared commands
and recording evidence. **Compare** means normalizing two capsules and
checking structural or evidence equivalence. The meta-generator does not
own domain physics, domain validation data, or scientific acceptance
thresholds.

### 2. Tiling Tree

**Split 1 - artifact form.**

- Human guide only.
- Machine-readable capsule only.
- Machine-readable capsule plus rendered guide.

The branches cover the user interface surface: human-only, automation-only,
or both. The chosen branch is both, because automation catches drift and
the rendered guide serves independent actors. Human-only has lower
maintenance cost initially but weak correctness guarantees. Automation-only
has stronger checks but higher reviewer load for humans auditing a run.

**Split 2 - execution depth.**

- Collect metadata only.
- Collect plus dry-run validation.
- Collect, dry-run, and execute checks.
- Collect, dry-run, execute, and compare regenerated capsules.

The branches cover increasing operational depth and do not overlap because
each branch strictly adds behavior. The chosen design names all four but
starts with collect plus dry-run and structural comparison. Full execution
and evidence comparison have the strongest evidence but the largest
operational blast radius when network, data, or hardware dependencies are
unavailable.

**Split 3 - ownership.**

- Platform owns all content and machinery.
- Applications own all generators independently.
- Platform owns generic machinery; applications own domain content.

The branches cover the platform/application allocation. The chosen branch
matches ADR-0014 and keeps scientific claims near the physics that owns
them. Platform-owning all content would violate the repository split.
Application-only generators would duplicate infrastructure and complicate
cross-scale workflows.

### 3. Concept Ownership Table

```text
Concept                  Owns                         Does not own
Meta-generator           collection, validation,       scientific truth of a
                         rendering, execution records domain claim
Capsule                  reproducibility contract      source archive or
                         and evidence index           container image
Source map               repository identities,        domain scope decisions
                         commits, remotes
Capability manifest      capability/provenance links   implementation code
Verification plan        commands and pass criteria    observational data
Validation plan          comparison products and       domain data curation
                         thresholds
Execution transcript     command results and hashes    reinterpretation of
                                                      failures as passes
Capsule comparison       normalization and divergence  scientific explanation
                         report                       for a failed check
Application repository   domain physics and data       platform schema rules
```

### 4. Real Workflow Stress Test

**Workflow A - platform dry-run before a PR.**

```text
cosmic-foundry capsule collect --target platform --output capsule.json
cosmic-foundry capsule dry-run capsule.json
cosmic-foundry capsule render capsule.json --output instructions.md
```

This checks that the platform's ADR index, status document,
environment recipe, schemas, formula register, replication specs, and
test selectors are internally consistent before a human reviews the PR.

**Workflow B - recursive platform comparison.**

```text
cosmic-foundry capsule collect --target platform --output capsule-a.json
cosmic-foundry capsule dry-run capsule-a.json
# Regenerate or check out the state described by capsule-a.json.
cosmic-foundry capsule collect --target platform --output capsule-b.json
cosmic-foundry capsule compare capsule-a.json capsule-b.json \
  --mode structural
```

This checks that the generated instructions reconstruct the same
nonvolatile platform contract after normalization.

**Workflow C - application capability execution.**

```text
cosmic-foundry capsule collect \
  --target application \
  --repo cosmic-foundry \
  --repo stellar-application \
  --capability C0007 \
  --output capsule.json
cosmic-foundry capsule execute capsule.json --evidence-dir evidence/
```

The capsule links the application capability spec to formulas,
derivations, externally grounded tests, platform schemas, validation
products, and provenance sidecars. Execution may skip hardware-specific
checks, but skipped requirements appear in the evidence summary.

### 5. Normalization Trace

**Platform dry-run trace.**

```text
author command
-> repository metadata, ADR index, status, schemas, replication specs
-> capsule sections with file references and declared commands
-> dry-run diagnostics for missing or stale references
-> rendered instructions and machine-readable failure report
```

**Recursive comparison trace.**

```text
capsule A plus regenerated state
-> capsule B collected from regenerated state
-> normalized capsule pair with volatile fields removed or canonicalized
-> structural or evidence divergence report
-> pass only if nonvolatile claims converge
```

**Application execution trace.**

```text
author command
-> platform source map plus application capability metadata
-> verification and validation command graph
-> execution transcript with exits, durations, hashes, skips, failures
-> evidence index and rendered independent-actor instructions
```

### 6. Ordering, Visibility, And Fences

Collection must precede dry-run because dry-run validates the collected
references. Dry-run should precede execution by default because execution
may be expensive and should not start from an obviously incomplete
capsule. Execution creates a materialization boundary: produced artifacts,
hashes, diagnostics, and transcripts are evidence and must not be silently
rewritten without a new run record. Compare creates a second
materialization boundary: a divergence report records exactly which
nonvolatile claims changed and which volatile fields were normalized.
Validation thresholds are visible inputs from application repositories,
not inferred by the platform.

### 7. Backend / Lowering Trace

A collector backend receives repository roots and target selectors and
returns normalized capsule sections. A renderer backend receives a valid
capsule and produces human-readable instructions. An executor backend
receives a validated capsule and runs declared commands in dependency
order. A comparison backend receives two capsules and normalization rules,
then returns structural or evidence equivalence plus a divergence report.
Backends may optimize filesystem scanning and command grouping, but they
may not drop declared checks, reinterpret skipped requirements as passes,
invent domain acceptance thresholds, or suppress nonvolatile divergence.

### 8. Alternative Failure Pass

The abstraction could collapse if capsules become another stale metadata
layer. The mitigation is dry-run failure on missing references and the
iteration rule that gaps should be fixed in authoritative artifacts.
The capsule could also secretly become a domain-specification language;
the ownership boundary forbids that by keeping domain scientific content
in application repositories. Full execution may serialize expensive work
unnecessarily; this is why dry-run is a first-class mode and execution is
allowed to be partial with explicit skips. The simpler manual-guide option
was rejected because its downstream correctness guarantees are weaker.
Recursive comparison could produce noisy failures if volatile fields are
underspecified; the mitigation is an explicit normalization policy and
schema-versioned comparison reports.

### 9. Decision Delta

Ready with named risks:

- Capsule schema churn is likely during the first implementation. Keep
  schema versions explicit and prefer additive changes while the first
  application repositories adopt the format.
- Full execution depends on hardware, data access, and application
  repositories that may not exist yet. Start with platform-only dry-run
  and treat skipped requirements as visible evidence, not success.
- Approximate idempotence depends on a careful volatile-field policy.
  Start with structural comparison, keep normalization rules explicit, and
  classify unexplained nonvolatile divergence as a defect.
