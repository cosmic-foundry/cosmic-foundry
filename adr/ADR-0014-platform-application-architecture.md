# ADR-0014 — Platform / Application Repository Architecture

## Context

Cosmic Foundry was originally scoped as a self-contained computational
astrophysics engine. As the project grows toward covering stellar physics,
cosmology, galactic dynamics, and planetary formation, two questions forced
an explicit decision:

1. Where does domain-specific physics code live — inside cosmic-foundry or
   in separate repositories?
2. Where does observational validation data live, and how does it relate
   to simulation code?

A parallel organizational decision dissolved `cosmic-observables` (a
separate repository that had accumulated SNIa observational data and
data-pipeline infrastructure). The dissolution forced a clean answer to
both questions.

## Decision

**Cosmic-foundry is the organizational platform.** It provides two
categories of capability:

1. **Computation infrastructure** — the Op / Region / Policy / Dispatch
   kernel abstraction, the mesh and field model, parallel I/O, global
   reductions, and the JAX backend. Every simulation domain depends on
   these regardless of its physics.

2. **Manifest and specification infrastructure** (`cosmic_foundry.manifests`)
   — the shared machinery for defining, validating, fetching, and tracking
   observational validation products and simulation specifications: HTTP
   client, `ValidationAdapter` protocol, `Provenance` dataclass, bibliography
   generator, base JSON schemas (catalog, validation-set, artifact-provenance),
   and schema validation utilities.

**Application repositories** (stellar-foundry, cosmological-foundry,
galactic-foundry, planetary-foundry, …) build on the platform. They are
thin on infrastructure — they do not reimplement mesh topology, dispatch,
I/O, provenance tracking, or schema validation — and may be rich on
physics. Each application repo provides:

- Physics implementations as cosmic-foundry Op / Policy instances.
- Domain-specific observational data: YAML manifests, adapters that
  implement `ValidationAdapter`, and the resulting artifacts.
- Domain-specific JSON schemas that extend the platform base schemas with
  domain vocabulary (unit enums, domain tags, additional required fields).
- Simulation specification manifests that reference target objects and
  validation products.

**Cross-scale workflows** that compose two or more application domains
(e.g., binary population synthesis spanning stellar and galactic scales)
live in their own repositories that depend on the relevant application
repos and the platform. Application repos do not depend on each other.

**The platform is intentionally heavy.** General-purpose infrastructure
lives in cosmic-foundry even when a single application would not need all
of it. The manifest infrastructure is an optional extra
(`pip install cosmic-foundry[observational]`) so that pure-simulation
users do not pull in HTTP and YAML dependencies.

## Consequences

- **Positive:** Application repos have a stable, tested foundation to build
  on. Shared infrastructure — especially provenance tracking and the
  `ValidationAdapter` protocol — is defined once and conforms uniformly
  across all domains.
- **Positive:** Cross-scale workflows are possible without coupling
  application repos to each other; they depend on the platform and the
  relevant application repos, not on each other.
- **Positive:** Breaking changes to the platform are visible and
  negotiable; they are not buried inside a single monorepo.
- **Negative:** Changes that span the platform/application boundary require
  coordinated PRs across repositories. This cost is accepted as the
  price of the separation.
- **Neutral:** The comparison-result schema — the contract between a
  simulation run and a validation product — lives in the platform, since
  all application repos must produce interoperable comparison outputs.
  Its design is deferred to Epoch 3 and recorded in a follow-up ADR.

## Alternatives considered

**Single monorepo.** All physics, all observational data, and all
workflows in one repository. Avoids cross-repo coordination. Rejected
because it conflates computation infrastructure (which every domain
shares) with domain physics (which is specific), making the codebase
harder to review and the dependency graph harder to manage.

**cosmic-observables as a permanent shared data repo.** Keep
`cosmic-observables` alongside `cosmic-foundry` as a sibling repo
serving all application repos. Rejected because the data-pipeline
infrastructure (HTTP client, provenance, adapter protocol) is platform
infrastructure that belongs with the platform, while domain-specific
observational data belongs with the application repo that uses it. A
separate data repo adds a coordination layer without a clear benefit
once the multi-repo application architecture is committed to.

**Thin platform, physics in applications only.** Platform provides only
computation primitives; all physics implementations live in application
repos. Rejected because it prevents cross-domain physics reuse (e.g.,
Newtonian hydrodynamics is needed by stellar-foundry, cosmological-foundry,
and galactic-foundry; implementing it three times would be incorrect).
