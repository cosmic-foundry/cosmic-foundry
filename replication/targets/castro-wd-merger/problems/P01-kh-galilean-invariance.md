# Problem: Kelvin–Helmholtz Galilean invariance

- **ID:** P01
- **Status:** Proposed
- **Target code:** CASTRO (version TBD; pin in
  `../golden/manifest.yaml` before fixtures are generated)
- **References:**
  - Katz et al. 2016, §4.3.1, arXiv:1512.06099
  - Robertson et al. 2010 — smoothed-interface KH setup
- **Capabilities required:** [C0001, C0002, C0003, C0004]

## Setup

Robertson et al. 2010 smoothed KH setup: two fluid layers with a
density contrast, a tanh-smoothed shear interface, and a
sinusoidal transverse-velocity perturbation seeded at a fixed
wavelength. Periodic boundaries, 2-D Cartesian domain, gamma-law
EOS (γ = 5/3). The initial state is evolved in the rest frame and
repeated in ≥2 bulk-velocity-boosted frames.

Robertson established that Eulerian codes *lose* KH growth in
boosted frames when the interface is a tangential discontinuity;
smoothed ICs recover invariance. Katz et al. 2016 show CASTRO
passes this test on its unsplit-PPM AMR solver.

We replicate the **scientific conclusion** — growth rate matches
linear theory within tolerance, invariant across boost frames
within a tighter tolerance — not CASTRO's bitstream. Any Euler
scheme that qualifies is acceptable; the replication is of the
test outcome, not the method.

## Success criterion

- Linear-mode KH growth rate at fiducial resolution agrees with
  linear theory within tolerance [TBD].
- Growth rates across ≥2 boost frames agree with one another
  within a tighter tolerance [TBD], demonstrating Galilean
  invariance.
- Measured convergence order on a refinement sequence is
  consistent with the underlying scheme's formal order.

## Verification plan

- Unit fixtures: seeded-mode amplitude vs. time at fiducial
  resolution, one fixture per boost frame.
- Convergence test: refinement sequence reporting measured
  growth rate per resolution; slope check on error vs. linear
  theory.
- Target-specific diagnostic: Fourier-amplitude extraction of
  the seeded transverse-velocity mode with exponential fit over
  the linear phase, supplied by C0004.

## Out of scope

Non-smoothed-interface KH. Compressibility / Mach-number sweeps.
Gravity-stratified or rotating variants. AMR behavior under KH
(the test is at uniform resolution per boost frame).

## Open questions

- Which hydro scheme does the engine ship as fiducial? CASTRO
  uses unsplit PPM + CTU; any scheme sufficient for passing
  invariance on smoothed ICs is acceptable. The engine-level
  choice is a separate decision (likely an ADR).
- Exact boost velocities used in §4.3.1; pin from the paper.
- Tolerance values on growth-rate match and invariance. Set
  after reading Robertson 2010's inter-scheme spread.
- Whether to pin and run a specific CASTRO release as an
  independent cross-check, or take Katz 2016's published result
  as the numerical reference.
