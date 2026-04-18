# Epoch 6 — Self-gravity and N-body

> Part of the [Cosmic Foundry roadmap](../index.md).

Close the loop with gravitational dynamics:

- Geometric multigrid Poisson on the AMR hierarchy (JAX where
  possible; Taichi fallback for irregular smoothers).
- Particle infrastructure — cell-in-cloud deposition, neighbor
  search — on Warp or Taichi.
- Barnes–Hut tree gravity with Ewald summation for periodic BCs.
- FMM prototype.

**Exit criterion:** Zel'dovich pancake and hydrostatic-equilibrium
tests match reference solutions, and a Zel'dovich-pancake live
explainer ships to the gallery with linked phase-space and
density-field views.
