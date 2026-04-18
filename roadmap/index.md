# Cosmic Foundry Roadmap Index

Cosmic Foundry's roadmap is split onto two separate planes:

- [Object-level roadmap](object-level.md) — what platform and simulation
  capabilities the codebase is building.
- [Meta-level roadmap](meta-level.md) — how the project verifies,
  validates, regenerates, and audits object-level claims.

The split is intentional. Object-level planning should not absorb the
mechanics of verification and reproducibility, and meta-level planning
should not decide scientific or computational scope.

## Track Selection

Each PR declares which track it advances:

- **Object-level** — engine, platform-service, application-facing, or
  physics capability work.
- **Meta-level** — reproducibility, verification, validation, provenance,
  evidence, or review-discipline work.
- **Both** — a change that intentionally moves an object-level capability
  and its evidence machinery together.

Cross-track dependencies must be explicit in the relevant PR description
and status update. PR #93 advances the meta-level track by defining the
reproducibility meta-generator and its first platform convergence plan;
the object-level track remains at Epoch 2 item #5.

## Current Track Positions

- **Object-level:** Epoch 2 — Mesh and AMR — in progress. Next listed
  item is task-graph driver — single-rank.
- **Meta-level:** M3 — Reproducibility meta-generator — planned. Next
  listed item is the platform-only capsule convergence slice.

See [STATUS.md](../STATUS.md) for the session-start summary.

## Detailed Plans

- [Object-level roadmap](object-level.md)
- [Meta-level roadmap](meta-level.md)
- [M3 reproducibility meta-generator plan](reproducibility-meta-generator.md)
- [Epoch 0 — Bootstrap](epoch-00-bootstrap.md)
- [Epoch 1 — Kernels](epoch-01-kernels.md)
- [Epoch 2 — Mesh and AMR](epoch-02-mesh.md)
- [Epoch 3 — Platform Services](epoch-03-platform-services.md)
- [Epoch 4 — Visualization](epoch-04-visualization.md)
- [Epoch 5 — Newtonian Hydro](epoch-05-newtonian-hydro.md)
- [Epoch 6 — Gravity and N-body](epoch-06-gravity-nbody.md)
- [Epoch 7 — Microphysics](epoch-07-microphysics.md)
- [Epoch 8 — MHD](epoch-08-mhd.md)
- [Epoch 9 — Radiation](epoch-09-radiation.md)
- [Epoch 10 — Relativistic](epoch-10-relativistic.md)
- [Epoch 11 — Particle Cosmology](epoch-11-particle-cosmology.md)
- [Epoch 12 — Moving Mesh](epoch-12-moving-mesh.md)
- [Epoch 13 — Stellar Evolution](epoch-13-stellar-evolution.md)
- [Epoch 14 — Subgrid Observables](epoch-14-subgrid-observables.md)
