# Epoch 5 — Newtonian hydrodynamics

> Part of the [Cosmic Foundry roadmap](index.md).

The first physics module and the template for every subsequent
one:

- Finite-volume Godunov with PPM reconstruction (SymPy-derived
  stencils).
- Riemann solvers: HLLC, HLLE, Roe.
- CFL time-stepping and passive scalars.
- Golden regression suite: Sod, Sedov, Noh, blast wave,
  Kelvin–Helmholtz, Rayleigh–Taylor.

**Exit criterion:** the standard hydro test battery matches
reference solutions across all kernel backends, and a Sod
shock-tube explainer page (live slider over γ and initial
conditions, unit-labeled axes, perceptual colormap) ships to
the public gallery as the first physics demo.
