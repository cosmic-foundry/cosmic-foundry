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

## Capabilities consumed

Union across the problems above:

- [C0001 — compressible Euler (Cartesian)](../../../capabilities/C0001-compressible-euler-cartesian.md)
- [C0005 — constant acceleration source](../../../capabilities/C0005-constant-acceleration-source.md)
- [C0006 — Helmholtz-family EOS](../../../capabilities/C0006-helmholtz-eos.md)
- [C0007 — aprox reaction network](../../../capabilities/C0007-aprox-reaction-network.md)
- [C0008 — Strang hydro–burn coupling](../../../capabilities/C0008-strang-burn-hydro-coupling.md)
- [C0009 — ignition-event diagnostic](../../../capabilities/C0009-ignition-event-diagnostic.md)

## Roadmap interaction

This target requires microphysics before MHD. The roadmap
reorder merged alongside this layout change delivers
microphysics at Epoch 6 and MHD at Epoch 7.
