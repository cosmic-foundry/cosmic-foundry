# Epoch 10 — Particle / meshless hydrodynamics and cosmology

> Part of the [Cosmic Foundry roadmap](index.md).

The second mesh paradigm and the cosmological stack:

- SPH with modern pressure–energy and density-independent
  variants; neighbor loops on Taichi or Warp.
- Meshless finite-mass and finite-volume methods.
- Comoving FRW integrator.
- 2LPT initial conditions.
- On-the-fly FOF and SUBFIND-style halo finders.
- Light-cones and high-dynamic-range power-spectrum estimator.

**Exit criterion:** a cosmological box reproduces a published
reference at fixed resolution within documented tolerance, and a
cosmological-flythrough explainer ships to the gallery with a
light-cone tour and a live halo-mass-function panel.
