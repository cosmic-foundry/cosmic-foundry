# Target: CASTRO 1-D nuclear ignition (Katz & Zingale 2019)

- **Paper:** Katz & Zingale (2019), *Numerical Stability of
  Detonations in White Dwarf Simulations*, arXiv:1903.00132.
- **Target code:** CASTRO. Pin a specific released version in
  `golden/manifest.yaml` before fixtures are generated.
- **Scope:** the 1-D test problem introduced in the paper,
  inspired by white-dwarf collision conditions, arguing that the
  time and location of initial thermonuclear ignition are
  converged only when spatial resolution is far below 1 km in
  the burning region.

## Problems

In order:

1. [P01 — 1-D nuclear ignition convergence](problems/P01-1d-nuclear-ignition.md)

## Maps required

Union across the problems above. Each entry names the map (physics
operator) the problem depends on; the implementing module will carry a
`Map:` block with domain, codomain, operator, and convergence order.

- Compressible Euler equations (Cartesian) — ∂_t(ρ, ρ**v**, ρE) + ∇·F = 0
- Constant acceleration source — S_mom = ρg, S_energy = ρ**v**·g
- Helmholtz-family equation of state — thermodynamics from Helmholtz free energy F(ρ, T)
- Aprox reaction network — nuclear energy generation rates for C/O burning
- Strang hydro–burn coupling — operator-split integration of hydro and burn steps
- Ignition-event diagnostic — detect and locate thermonuclear runaway

## Roadmap interaction

This target requires microphysics before MHD. The roadmap
reorder merged alongside this layout change delivers
microphysics at Epoch 6 and MHD at Epoch 7.
