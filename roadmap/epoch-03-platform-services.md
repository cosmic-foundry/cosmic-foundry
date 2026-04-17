# Epoch 3 — Platform Services

> Part of the [Cosmic Foundry roadmap](index.md).

## Scope

Epoch 3 delivers the non-physics platform infrastructure that
application repositories depend on before they can produce or compare
simulation outputs against observational data. It also executes the
organizational transitions that the platform/application architecture
requires: dissolving `cosmic-observables` and bootstrapping
`stellar-foundry` as the first real application repo.

This epoch runs concurrently with the tail of Epoch 2 (multi-rank halo
fill and I/O work) and must close before the first application-side
physics PR opens in stellar-foundry.

---

## Deliverables

### 1. `cosmic_foundry.manifests` — platform manifest infrastructure

A new submodule providing the shared machinery that all application
repos use when defining, validating, and tracking observational and
simulation artifacts.

- **HTTP client** (`http_client.py`). Dual-identity fetch (research vs.
  bot) with `robots.txt` enforcement. Used by application-repo adapters
  to fetch upstream observational data. Migrated from `cosmic-observables`
  and generalized.
- **`ValidationAdapter` protocol**. The interface that every
  domain-repo adapter implements: fetch upstream source, normalize to
  platform-defined artifact schema, write artifact + provenance sidecar.
- **`Provenance` dataclass + sidecar writer** (`provenance.py`). Records
  adapter identity, upstream release pin, content hash, artifact path,
  and row count for both observational artifacts and simulation run
  outputs. General across both sides of the comparison contract.
- **Bibliography generator** (`bibliography.py`). Aggregates provenance
  records across a collection of artifacts into a human-readable
  bibliography. Migrated from `cosmic-observables`.
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
will be written in stellar-foundry, not here.

### 4. cosmic-observables dissolution

- Migrate `http_client.py` and `bibliography.py` to
  `cosmic_foundry.manifests` (above).
- Migrate base JSON schemas to `cosmic_foundry/schemas/`.
- Replace the `cosmic-observables` README with an archive notice
  pointing to stellar-foundry (observational data) and cosmic-foundry
  (platform infrastructure).
- Archive the repo on GitHub.

### 5. stellar-foundry bootstrap

- `pyproject.toml`, CI, pre-commit configuration matching platform
  conventions.
- `AI.md` delegating to cosmic-foundry's rules with stellar-specific
  additions (fork/PR targets, environment notes).
- Package skeleton: `src/stellar_foundry/__init__.py`.
- `adr/README.md` stub.
- `STATUS.md`.
- Directory scaffold for `src/stellar_foundry/physics/` (empty —
  placeholder for future stellar physics implementations) and
  `src/stellar_foundry/validation/sne_ia/` (receives the SNIa
  observational content).

### 6. SNIa observational content migration to stellar-foundry

- Migrate `adapters/` (pantheon_plus, csp_dr3, foundation, tns) to
  `src/stellar_foundry/validation/sne_ia/adapters/`, updating imports
  and implementing the `ValidationAdapter` protocol from
  `cosmic_foundry.manifests`.
- Migrate `alias_table.py` and `cross_match.py` to
  `src/stellar_foundry/validation/sne_ia/`.
- Migrate `observables/sne-ia/` YAML manifests (catalogs,
  validation-sets, objects, filters, filter-matches) to
  `stellar-foundry/observables/sne-ia/`.
- Migrate domain-specific JSON schemas (`photometry`, `filter`,
  `filter-match`, `object`) to `stellar-foundry/schemas/`.
- Migrate artifacts to `stellar-foundry/artifacts/sne-ia/`.
- Adapt and migrate ADR-0001, ADR-0002, ADR-0003 from
  `cosmic-observables/adr/` to `stellar-foundry/adr/`,
  recontextualized for their new home.
- Migrate and update tests.

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

This ADR also records the dissolution of `cosmic-observables` and the
reasoning behind the split.

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
- `cosmic-observables` is archived on GitHub.
- stellar-foundry CI is green; the SNIa observational content is present,
  tested, and the adapters implement the `ValidationAdapter` protocol.
- A reviewer can trace a Pantheon+ distance modulus from the upstream
  source through the stellar-foundry adapter to the artifact provenance
  sidecar, using only cosmic-foundry manifest infrastructure.

---

## Sequencing notes

Deliverables 1–3 (platform side) should land first as cosmic-foundry
PRs, so that stellar-foundry bootstrap (4–6) can depend on a released
or locally-installed version of the new `[observational]` extra. In
practice, editable installs make the ordering flexible, but the
`ValidationAdapter` protocol must be defined before the migrated adapters
are written.

Deliverable 5 (stellar-foundry bootstrap) can proceed in parallel with
Deliverable 1 once the protocol interface is sketched, even before the
full platform-side implementation is merged.
