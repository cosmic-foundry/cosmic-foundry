# Epoch 7 — Microphysics sub-layer

> Part of the [Cosmic Foundry roadmap](index.md).

Bring up the equations of state and reaction networks that later
physics modules depend on:

- Abstract EOS interface with ideal-gas, Helmholtz, piecewise
  polytropic, and tabulated nuclear finite-T implementations.
  Tables are JAX-jittable piecewise interpolants.
- Reaction-network engine with autodiff-generated Jacobians,
  α-network reference, and a path to large adaptive networks.
- Primordial and metal cooling tables.
- Radiation opacities.

**Exit criterion:** thermonuclear flame and primordial cooling
benchmarks match published results, and a 1-D thermonuclear-flame
explainer ships with interactive EOS / network switching.
