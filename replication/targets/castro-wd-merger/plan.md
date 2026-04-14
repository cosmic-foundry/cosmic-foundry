# Target: CASTRO white-dwarf merger methodology (Katz et al. 2016)

- **Paper:** Katz, Zingale, Calder, Swesty, Almgren, Zhang (2016),
  *White Dwarf Mergers on Adaptive Meshes I. Methodology and Code
  Verification*, arXiv:1512.06099.
- **Target code:** CASTRO. Pin a specific released version in
  `golden/manifest.yaml` before fixtures are generated.
- **Scope:** §4.3.1 only — the Kelvin–Helmholtz Galilean-
  invariance test, following Robertson et al. (2010)'s smoothed-
  interface setup. Other parts of the paper (self-gravity,
  rotation, WD-merger integration) are out of scope here and
  would be added as separate targets when pursued.

## Problems

In order:

1. [P01 — KH Galilean invariance](problems/P01-kh-galilean-invariance.md)

## Capabilities consumed

Union across the problems above:

- [C0001 — compressible Euler (Cartesian)](../../capabilities/C0001-compressible-euler-cartesian.md)
- [C0002 — gamma-law EOS](../../capabilities/C0002-gamma-law-eos.md)
- [C0003 — passive-scalar advection](../../capabilities/C0003-passive-scalar-advection.md)
- [C0004 — linear-mode growth-rate diagnostic](../../capabilities/C0004-linear-mode-diagnostic.md)
