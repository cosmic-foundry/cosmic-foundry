# Target: CASTRO white-dwarf merger methodology (Katz et al. 2016)

- **Paper:** Katz, Zingale, Calder, Swesty, Almgren, Zhang (2016),
  *White Dwarf Mergers on Adaptive Meshes I. Methodology and Code
  Verification*, arXiv:1512.06099.
- **Target code:** CASTRO. Pin a specific released version in
  `golden/manifest.yaml` before we generate fixtures.
- **Scope of this target:** §4.3.1 only — the Kelvin–Helmholtz
  Galilean-invariance test, following the smoothed-interface
  setup of Robertson et al. (2010). Other parts of the paper
  (self-gravity, rotation, WD-merger integration) are out of
  scope here and would be added as separate targets when
  pursued.

## What we replicate

The §4.3.1 test verifies that the Euler solver preserves the
linear KH growth rate under bulk Galilean boosts when the shear
interface is smoothed (per Robertson et al. 2010) rather than a
tangential discontinuity. Robertson established that Eulerian
codes *lose* KH growth in boosted frames with discontinuous ICs;
smoothed ICs recover invariance. Katz et al. show CASTRO passes
this test on its unsplit-PPM AMR solver.

We reproduce the **scientific conclusion** — growth rate matches
linear theory within tolerance, invariant across boost frames to
within a tighter tolerance — not CASTRO's bitstream. Any Euler
scheme that qualifies is acceptable; the replication is of the
test outcome, not the method.

## Success criterion

- Linear-mode KH growth rate at fiducial resolution agrees with
  linear theory within tolerance [TBD].
- Growth rates across ≥2 boosted frames agree with one another
  within a tighter tolerance [TBD], demonstrating Galilean
  invariance.
- Measured convergence order on a refinement sequence is
  consistent with the underlying scheme's formal order.

## Capability checklist

Each item becomes a spec under `specs/` as it is tackled:

1. Compressible Euler solver, 2-D Cartesian, periodic BCs,
   gamma-law EOS (γ = 5/3).
2. Passive-scalar (tracer) advection, for mode-amplitude
   diagnostics that track the shear layer under boost.
3. Linear-mode growth-rate measurement: extract the Fourier
   amplitude of the seeded transverse-velocity mode and fit an
   exponential over the linear phase.

Capabilities 1 and 2 are expected to be shared with the CASTRO
detonation target (`replication/castro-detonation/`). Under the
current per-target `specs/` layout, a shared capability lives in
whichever target's `specs/` first writes it; other targets
cross-reference rather than duplicating. If that convention
proves awkward once either target's specs are written, revisit
the workflow (see `replication/README.md` "Living-spec
discipline").

## Open questions

- Which hydro scheme does the engine ship as fiducial? CASTRO
  uses unsplit PPM + CTU; any scheme sufficient for passing
  invariance on smoothed ICs is acceptable for this test. The
  engine-level choice is a separate decision (likely an ADR).
- Exact boost velocities used in §4.3.1; pin from the paper
  when the first capability spec is written.
- Tolerance values on growth-rate match and invariance. Set
  after reading Robertson 2010's inter-scheme spread.
- Whether to pin and run a specific CASTRO release as an
  independent cross-check, or take Katz 2016's published result
  as the numerical reference.
