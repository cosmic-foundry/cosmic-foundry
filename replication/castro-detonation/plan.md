# Target: CASTRO 1-D nuclear detonation (Katz & Zingale 2019)

- **Paper:** Katz & Zingale (2019), *Numerical Stability of
  Detonations in White Dwarf Simulations*, arXiv:1903.00132.
- **Target code:** CASTRO. Pin a specific released version in
  `golden/manifest.yaml` before we generate fixtures.
- **Scope of this target:** the 1-D test problem introduced in
  the paper, inspired by white-dwarf collision conditions, used
  to argue that the **time and location of initial thermonuclear
  ignition** are converged only when spatial resolution is far
  below 1 km in the burning region. Coarser resolution yields
  ignition times or ignition positions that are numerical
  artifacts rather than physical predictions.

## What we replicate

A 1-D planar setup: hot, dense, carbon-rich upstream conditions
approximating a colliding-WD regime; a domain long enough to
contain the pre-ignition evolution; operator-split coupling of
hydro and a reaction network (e.g. aprox13) via a degenerate EOS
(Helmholtz-family). The test evolves the initial state forward
and records the first self-sustained thermonuclear ignition
event — its time and spatial location.

The headline finding is a resolution study: the measured
ignition time and location converge only below ~1 km cell size;
above that threshold they are sensitive to resolution in a way
that indicates numerical rather than physical ignition.

We reproduce the **scientific conclusion** — the resolution
threshold at which ignition time and location stabilize to
within a stated tolerance — not CASTRO's bitstream. Our burner
and EOS need not be bit-identical to CASTRO's; they must show
the same ignition-convergence behavior on the same ICs.

## Success criterion

- Ignition time t_ign and ignition location x_ign converge
  monotonically as resolution is refined through the grid
  sequence specified in the paper.
- The resolution at which relative changes in t_ign and x_ign
  drop below tolerance [TBD] matches the paper's sub-1 km
  finding within a factor of [TBD].
- Energy and species-abundance conservation invariants hold to
  within stated tolerances at every resolution, so that an
  "ignition" flagged at coarse resolution is not masking a
  conservation failure.

## Capability checklist

Each item becomes a spec under `specs/` as it is tackled:

1. Compressible Euler solver, 1-D planar, specified boundary
   conditions (outflow / sustained upstream — confirm from
   paper).
2. Constant acceleration source term representing gravity: a
   fixed uniform body force on the momentum equation and the
   associated work term on the energy equation. The paper uses
   this in place of a full self-gravity solve; its magnitude
   and direction must be pinned from the paper, along with any
   hydrostatic-balance requirement on the initial pressure
   profile that drives the well-balanced-discretization choice.
3. Degenerate stellar EOS, Helmholtz-family, with the
   thermodynamic derivatives required by the hydrodynamics and
   burner.
4. Nuclear reaction network (aprox13 expected; confirm exact
   network from paper) with an implicit integrator.
5. Strang-split hydro–burn coupling, including the reactive
   source treatment used by the target.
6. Ignition-event diagnostic: detect and time-stamp the onset
   of self-sustained thermonuclear burning, and record its
   spatial location. The detection criterion itself (local
   energy generation rate vs. loss, species threshold, or a
   combination) is load-bearing for this target's success
   criterion and must be pinned from the paper.

Capability 1 is expected to be shared with the KH target
(`replication/castro-wd-merger/`). Under the current per-target
`specs/` layout, the shared capability lives in whichever
target's `specs/` first writes it; the other cross-references
rather than duplicating. Capability 2 (constant acceleration
source) is not needed by the KH test but is a general-enough
hydro source term that a future target may reuse it, so it is
written as a standalone spec rather than baked into the Euler
spec. Capabilities 3–6 are unique to this target at present.

## Open questions

- Exact reaction network (aprox13 vs aprox19 vs other). Pin
  from the paper when the reaction-network spec is written.
- Upstream thermodynamic state, domain extent, and termination
  time. Pin from the paper.
- Magnitude and direction of the constant gravitational
  acceleration, and whether the initial profile is required to
  be in hydrostatic balance against it (which constrains the
  discretization choice toward a well-balanced scheme).
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

## Roadmap interaction

Replicating this target requires microphysics (EOS + reaction
network) earlier in the implementation order than MHD. A
follow-up roadmap PR reorders the epochs accordingly; this plan
is a load-bearing input to that reorder.
