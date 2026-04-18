# Architectural Decision Records

Each ADR captures one architectural decision: the context that forced it,
the choice made, the consequences, and the alternatives considered. ADRs
describe current architecture. When a decision changes, edit the ADR in
place — git history records what changed and when. If a decision is
entirely withdrawn, remove it from the registry. See
[ADR-0005 §Decision → ADR editing policy](meta-level/ADR-0005-branch-pr-attribution-discipline.md#adr-editing-policy)
for the authoritative rule.

## How To Use This Registry

- **On startup**, read this registry, then read the architecture plane
  relevant to the work.
- **When work touches a topic listed here**, read the full ADR before
  making changes. The registry and plane documents are pointers, not
  summary substitutes.
- **When making a new architectural decision**, copy
  `adr-template.md` to `ADR-NNNN-<short-title>.md`, add a line to this
  registry, and add the ADR to exactly one architecture plane unless the
  decision genuinely needs a bridge note in both.

## Architecture Planes

- [Object-level architecture](object-level/README.md) — what the platform and
  application repositories are: execution model, kernels, mesh,
  diagnostics, I/O, visualization, manifests, repository boundaries, and
  eventually physics capabilities.
- [Meta-level architecture](meta-level/README.md) — how the project verifies and
  regenerates object-level claims: PR discipline, replication workflow,
  derivation lanes, numerical-transcription discipline, validation
  evidence, and reproducibility capsules.

## ADR Registry

The registry is grouped by architecture plane so the CI index checker can
verify every ADR file is listed here while detailed track-specific
architecture stays on its own plane.

### Object-Level ADRs

| ADR | Primary concern |
|-----|-----------------|
| [ADR-0001](object-level/ADR-0001-python-with-runtime-codegen.md) | Python-only engine with runtime code generation |
| [ADR-0002](object-level/ADR-0002-jax-primary-kernel-backend.md) | JAX + XLA as primary kernel backend |
| [ADR-0003](object-level/ADR-0003-jax-distributed-host-parallelism.md) | `jax.distributed` host parallelism |
| [ADR-0004](object-level/ADR-0004-sphinx-myst-docs-stack.md) | Sphinx + MyST-NB documentation stack |
| [ADR-0006](object-level/ADR-0006-visualization-stack.md) | Visualization and science-communication stack |
| [ADR-0009](object-level/ADR-0009-float64-default-precision.md) | Float64 default precision |
| [ADR-0010](object-level/ADR-0010-kernel-abstraction-model.md) | Kernel abstraction model |
| [ADR-0011](object-level/ADR-0011-halo-fill-fence.md) | Halo fill fence |
| [ADR-0012](object-level/ADR-0012-global-reduction-primitive.md) | Global reduction primitive |
| [ADR-0014](object-level/ADR-0014-platform-application-architecture.md) | Platform / application repository architecture |

### Meta-Level ADRs

| ADR | Primary concern |
|-----|-----------------|
| [ADR-0005](meta-level/ADR-0005-branch-pr-attribution-discipline.md) | Branch, PR, history, and attribution discipline |
| [ADR-0007](meta-level/ADR-0007-replication-workflow.md) | Bounded-increment replication workflow |
| [ADR-0008](meta-level/ADR-0008-numerical-transcription-discipline.md) | Numerical-transcription discipline stub |
| [ADR-0013](meta-level/ADR-0013-derivation-first-lane.md) | Derivation-first lane for physics capabilities |
| [ADR-0015](meta-level/ADR-0015-reproducibility-meta-generator.md) | Reproducibility meta-generator |
