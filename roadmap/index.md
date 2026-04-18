# Cosmic Foundry Roadmap Index

Cosmic Foundry's roadmap is split onto two separate planes:

- [Object-level roadmap](object-level/README.md) — what platform and simulation
  capabilities the codebase is building.
- [Meta-level roadmap](meta-level/README.md) — how the project verifies,
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

- [Object-level roadmap](object-level/README.md)
- [Meta-level roadmap](meta-level/README.md)
- [M3 reproducibility meta-generator plan](meta-level/reproducibility-meta-generator.md)
- [Epoch 0 — Bootstrap](object-level/epoch-00-bootstrap.md)
- [Epoch 1 — Kernels](object-level/epoch-01-kernels.md)
- [Epoch 2 — Mesh and AMR](object-level/epoch-02-mesh.md)
- [Epoch 3 — Platform Services](object-level/epoch-03-platform-services.md)
- [Epoch 4 — Visualization](object-level/epoch-04-visualization.md)
- [Epoch 5 — Newtonian Hydro](object-level/epoch-05-newtonian-hydro.md)
- [Epoch 6 — Gravity and N-body](object-level/epoch-06-gravity-nbody.md)
- [Epoch 7 — Microphysics](object-level/epoch-07-microphysics.md)
- [Epoch 8 — MHD](object-level/epoch-08-mhd.md)
- [Epoch 9 — Radiation](object-level/epoch-09-radiation.md)
- [Epoch 10 — Relativistic](object-level/epoch-10-relativistic.md)
- [Epoch 11 — Particle Cosmology](object-level/epoch-11-particle-cosmology.md)
- [Epoch 12 — Moving Mesh](object-level/epoch-12-moving-mesh.md)
- [Epoch 13 — Stellar Evolution](object-level/epoch-13-stellar-evolution.md)
- [Epoch 14 — Subgrid Observables](object-level/epoch-14-subgrid-observables.md)
