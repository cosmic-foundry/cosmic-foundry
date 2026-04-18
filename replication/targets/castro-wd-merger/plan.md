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

## Maps required

Union across the problems above. Each entry names the map (physics
operator) the problem depends on; the implementing module will carry a
`Map:` block with domain, codomain, operator, and convergence order.

- Compressible Euler equations (Cartesian) — ∂_t(ρ, ρ**v**, ρE) + ∇·F = 0
- Gamma-law equation of state — P = (γ−1)ρe
- Passive-scalar advection — ∂_t(ρX) + ∇·(ρ**v**X) = 0
- Linear-mode growth-rate diagnostic — projection onto eigenmode amplitude
