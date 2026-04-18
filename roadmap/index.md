# Cosmic Foundry Roadmap Index

Cosmic Foundry's roadmap is organized into two tracks:

- [Implementation roadmap](implementation/README.md) — **Track 1:** how the
  codebase delivers physics capabilities. Epoch sequencing and platform
  delivery. What the engine computes is expressed by the maps in the code,
  each documented with a `Map:` block stating domain, codomain, operator,
  and approximation parameters (ADR-0016).
- [Verification roadmap](verification/README.md) — **Track 2:** how Track 1
  claims are verified to be correctly realized. Reproducibility, convergence
  testing, replication targets, and validation evidence.

The tracks are intentionally separate. Track 1 sequences delivery without
absorbing verification mechanics. Track 2 verifies without deciding
scientific or computational scope.

## Track Selection

Each PR declares which track it advances:

- **Implementation** — engine, platform-service, or application-facing code.
- **Verification** — reproducibility, verification, validation, provenance,
  evidence, or review-discipline work.
- **Multiple tracks** — a change that intentionally advances both implementation
  and its verification evidence together.

Cross-track dependencies must be explicit in the relevant PR description
and status update.

## Current Track Positions

- **Implementation:** Epoch 2 — Mesh and AMR — in progress. Next item is
  task-graph driver — single-rank.
- **Verification:** M3 — Map: block coverage — ensure every implemented
  physics map carries a complete `Map:` block with domain, codomain,
  operator, Θ, and p.

See [STATUS.md](../STATUS.md) for the session-start summary.

## Detailed Plans

- [Implementation roadmap](implementation/README.md)
- [Verification roadmap](verification/README.md)
- [M3 verification plan](verification/reproducibility-meta-generator.md)
- [Epoch 0 — Bootstrap](implementation/epoch-00-bootstrap.md)
- [Epoch 1 — Kernels](implementation/epoch-01-kernels.md)
- [Epoch 2 — Mesh and AMR](implementation/epoch-02-mesh.md)
- [Epoch 3 — Platform Services](implementation/epoch-03-platform-services.md)
- [Epoch 4 — Visualization](implementation/epoch-04-visualization.md)
- [Epoch 5 — Newtonian Hydro](implementation/epoch-05-newtonian-hydro.md)
- [Epoch 6 — Gravity and N-body](implementation/epoch-06-gravity-nbody.md)
- [Epoch 7 — Microphysics](implementation/epoch-07-microphysics.md)
- [Epoch 8 — MHD](implementation/epoch-08-mhd.md)
- [Epoch 9 — Radiation](implementation/epoch-09-radiation.md)
- [Epoch 10 — Relativistic](implementation/epoch-10-relativistic.md)
- [Epoch 11 — Particle Cosmology](implementation/epoch-11-particle-cosmology.md)
- [Epoch 12 — Moving Mesh](implementation/epoch-12-moving-mesh.md)
- [Epoch 13 — Stellar Evolution](implementation/epoch-13-stellar-evolution.md)
- [Epoch 14 — Subgrid Observables](implementation/epoch-14-subgrid-observables.md)
