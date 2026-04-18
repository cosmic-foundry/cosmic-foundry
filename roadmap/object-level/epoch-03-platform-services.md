# Epoch 3 — Platform Services

> Part of the [Cosmic Foundry roadmap](../index.md).

## Scope

Epoch 3 delivers the non-physics platform infrastructure that
application repositories depend on before they can produce or compare
simulation outputs against observational data. It also executes the
organizational transitions that the platform/application architecture
requires: bootstrapping the first application repository and migrating
any observational pipeline content that was accumulated in predecessor
repositories into its proper home.

This epoch runs concurrently with the tail of Epoch 2 (multi-rank halo
fill and I/O work) and must close before the first application-side
physics PR opens.

---

## Deliverables

### 1. `cosmic_foundry.manifests` — platform manifest infrastructure

A new submodule providing the shared machinery that all application
repos use when defining, validating, and tracking observational and
simulation artifacts.

- **HTTP client** (`http_client.py`). Dual-identity fetch (research vs.
  bot) with `robots.txt` enforcement. Used by application-repo adapters
  to fetch upstream observational data.
- **`ValidationAdapter` protocol**. The interface that every
  domain-repo adapter implements: fetch upstream source, normalize to
  platform-defined artifact schema, write artifact + provenance sidecar.
- **`Provenance` dataclass + sidecar writer** (`provenance.py`). Records
  adapter identity, upstream release pin, content hash, artifact path,
  and row count for both observational artifacts and simulation run
  outputs. General across both sides of the comparison contract.
- **Bibliography generator** (`bibliography.py`). Aggregates provenance
  records across a collection of artifacts into a human-readable
  bibliography.
- **Base JSON schemas** (`schemas/`). Three domain-neutral schemas that
  application repos extend with domain-specific fields:
  - `catalog.schema.json` — upstream data source metadata (authority,
    access terms, release pin, URL, hash).
  - `validation-set.schema.json` — curated selection with stated
    scientific question, upstream catalog references, selection cuts,
    observables, units, and caveats.
  - `artifact-provenance.schema.json` — provenance sidecar format shared
    by observational artifacts and simulation run records.
- **Schema validation machinery**. A thin wrapper around `jsonschema`
  that validates manifests against the base schemas and any
  domain-supplied extension schemas, with clear error messages for
  manifest authors.
- **`[observational]` optional extra** in `pyproject.toml`. Adds
  `requests`, `jsonschema`, and `pyyaml` without affecting users who
  install cosmic-foundry for pure simulation use.

### 2. Comparison-result schema

The contract between a simulation run and a validation product:
simulation run ID, validation-product ID, observable, value, units,
tolerance, covariance handling, and provenance. Defined here in the
platform so that all application repos produce interoperable comparison
outputs and the eventual human explorer can display them uniformly.

Record the schema design as an ADR before this deliverable closes.

### 3. Simulation specification format

Resolution of the "Problem-setup surface" crossroads from the index:
YAML manifests validated against JSON schemas. A simulation specification
manifest carries the physics model identity, initial-condition parameters,
target object reference (linking to the application repo's object
manifests), and the validation products it intends to compare against.

Deliver a base `sim-spec.schema.json` and document the convention.
Record the decision as an ADR. The first concrete sim-spec manifests
will be written in application repos, not here.

### 4. Application repository bootstrapping

The platform-side work in deliverables 1–3 defines the contracts that
application repos implement. Alongside those platform PRs:

- Confirm the `ValidationAdapter` protocol, base schemas, and
  `[observational]` install surface are stable enough that an application
  repo can pin a version and build on them.
- Document the expected application repo layout (directory structure,
  `AI.md` delegation pattern, CI conventions) in the platform so that
  new application repos can be bootstrapped consistently.

The actual bootstrapping and domain-specific content migration happen
in the application repositories, not here.

---

## Design prerequisites

### Platform/application architecture ADR

Before opening any code PR in this epoch, record the platform/application
split as an ADR in cosmic-foundry:
- cosmic-foundry is the organizational platform; application repos are
  thin on infrastructure, rich on physics.
- What belongs in the platform (computation primitives, manifest
  machinery, comparison-result schema) vs. application repos (domain
  physics, domain-specific manifests and adapters, validation data).
- The pattern for cross-scale workflows (separate repo depending on
  multiple application repos and the platform).

### Comparison-result schema design

The comparison-result schema must be agreed before the migration PRs
open, because the `ValidationAdapter` protocol references it. Sketch
the schema (even informally) and record it before the first Epoch 3 code
PR.

---

## Exit criteria

- `cosmic_foundry.manifests` ships with tests and is importable from
  any application repo via `pip install cosmic-foundry[observational]`.
- The comparison-result and sim-spec schemas are defined and recorded in
  ADRs.
- The expected application repo layout and delegation pattern are
  documented in the platform.
- At least one application repo has been bootstrapped using the platform
  manifest infrastructure and its adapters implement `ValidationAdapter`.

---

## Sequencing notes

Deliverables 1–3 (platform side) should land first as cosmic-foundry
PRs, so that application repo bootstrap can depend on a released or
locally-installed version of the new `[observational]` extra. In
practice, editable installs make the ordering flexible, but the
`ValidationAdapter` protocol must be defined before domain adapters
are written.

Application repo bootstrapping (deliverable 4, application side) can
proceed in parallel with deliverable 1 once the protocol interface is
sketched, even before the full platform-side implementation is merged.
