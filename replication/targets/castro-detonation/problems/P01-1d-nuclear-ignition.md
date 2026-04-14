# Problem: 1-D nuclear ignition convergence

- **ID:** P01
- **Status:** Proposed
- **Target code:** CASTRO (version TBD; pin in
  `../golden/manifest.yaml` before fixtures are generated)
- **References:**
  - Katz & Zingale 2019, arXiv:1903.00132
- **Capabilities required:** [C0001, C0005, C0006, C0007,
  C0008, C0009]

## Setup

A 1-D planar setup with hot, dense, carbon-rich upstream
conditions approximating a colliding-WD regime; a computational
domain long enough to contain the pre-ignition evolution;
operator-split coupling of hydro and a reaction network (aprox13
expected; confirm from paper) via a degenerate EOS (Helmholtz-
family). A fixed uniform gravitational acceleration is applied
as a body force — the paper uses this in place of a self-gravity
solve. If the initial pressure profile must be in hydrostatic
balance against that acceleration, the Euler discretization
(C0001) must be well-balanced.

The test evolves the initial state forward and records the first
self-sustained thermonuclear ignition event: its time t_ign and
spatial location x_ign. The paper's headline is that t_ign and
x_ign converge only below ~1 km cell size; coarser resolution
yields numerical rather than physical ignition.

We replicate the **scientific conclusion** — the resolution
threshold at which t_ign and x_ign stabilize to within stated
tolerance — not CASTRO's bitstream. Our burner and EOS need not
be bit-identical to CASTRO's; they must show the same
ignition-convergence behavior on the same ICs.

## Success criterion

- t_ign and x_ign converge monotonically as resolution is
  refined through the grid sequence specified in the paper.
- The resolution at which relative changes in t_ign and x_ign
  drop below tolerance [TBD] matches the paper's sub-1 km
  finding within a factor of [TBD].
- Energy and species-abundance conservation invariants hold to
  within stated tolerances at every resolution, so that an
  "ignition" flagged at coarse resolution is not masking a
  conservation failure.

## Verification plan

- Unit fixtures: (t_ign, x_ign) per resolution in the paper's
  grid sequence.
- Convergence test: the sequence itself — relative change in
  t_ign and x_ign under each halving is the measured quantity,
  with the paper's sub-1 km threshold as the reference.
- Target-specific diagnostics: ignition-event detector (C0009)
  returning (t_ign, x_ign); total-energy and total-species-mass
  conservation checks per timestep.

## Out of scope

Multidimensional detonations. Realistic self-gravity. Ignition
mechanisms other than the paper's setup.

## Open questions

- Exact reaction network (aprox13 vs aprox19 vs other). Pin
  from the paper.
- Upstream thermodynamic state, domain extent, and termination
  time. Pin from the paper.
- Magnitude and direction of the constant gravitational
  acceleration, and whether the initial profile is required to
  be in hydrostatic balance against it (which constrains C0001
  toward a well-balanced discretization).
- Specific EOS implementation (Timmes Helmholtz table, or an
  equivalent). The paper's choice is the pinned target; our
  implementation may differ in algorithmic detail so long as
  convergence behavior reproduces.
- Tolerance on "converged ignition" — relative change in t_ign
  and x_ign under a resolution halving, below which we call
  convergence. Pin from the paper's reported grid sequence.
- Exact ignition-detection criterion used in the paper. Without
  pinning this, the "same test" claim is brittle.
- Whether the engine-level EOS and network choices motivate a
  follow-on ADR (likely yes; no ADR presently covers
  microphysics).
