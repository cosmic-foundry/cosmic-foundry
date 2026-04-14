# Target: CASTRO 1-D nuclear detonation (Katz & Zingale 2019)

- **Paper:** Katz & Zingale (2019), *Numerical Stability of
  Detonations in White Dwarf Simulations*, arXiv:1903.00132.
- **Target code:** CASTRO. Pin a specific released version in
  `golden/manifest.yaml` before we generate fixtures.
- **Scope of this target:** the 1-D detonation test problem
  introduced in the paper, inspired by white-dwarf collision
  conditions, used to argue that converged thermonuclear
  ignition requires spatial resolution far below 1 km in the
  burning region.

## What we replicate

A 1-D planar thermonuclear detonation: hot, dense, carbon-rich
upstream conditions; a computational domain long enough to
resolve the reaction zone; operator-split coupling of hydro and
a reaction network (e.g. aprox13) via a degenerate EOS
(Helmholtz-family). The headline finding is a resolution study:
converged detonation speed and post-shock structure emerge only
below ~1 km cell size in the burning region.

We reproduce the **scientific conclusion** — the resolution
threshold at which detonation speed and structure stabilize to
within a stated tolerance — not CASTRO's bitstream. Our burner
and EOS need not be bit-identical to CASTRO's; they must show
the same convergence behavior on the same ICs.

## Success criterion

- Detonation speed and post-shock thermodynamic state converge
  monotonically as resolution is refined through the grid
  sequence specified in the paper.
- The resolution at which relative change in detonation speed
  drops below tolerance [TBD] matches the paper's finding
  within a factor of [TBD].
- Energy and species-abundance conservation invariants hold to
  within stated tolerances at every resolution.

## Capability checklist

Each item becomes a spec under `specs/` as it is tackled:

1. Compressible Euler solver, 1-D planar, specified boundary
   conditions (outflow / sustained upstream — confirm from
   paper).
2. Degenerate stellar EOS, Helmholtz-family, with the
   thermodynamic derivatives required by the hydrodynamics and
   burner.
3. Nuclear reaction network (aprox13 expected; confirm exact
   network from paper) with an implicit integrator.
4. Strang-split hydro–burn coupling, including the reactive
   source treatment used by the target.

Capability 1 is expected to be shared with the KH target
(`replication/castro-wd-merger/`). Under the current per-target
`specs/` layout, the shared capability lives in whichever
target's `specs/` first writes it; the other cross-references
rather than duplicating. Capabilities 2–4 are unique to this
target at present.

## Open questions

- Exact reaction network (aprox13 vs aprox19 vs other). Pin
  from the paper when the reaction-network spec is written.
- Upstream thermodynamic state, domain extent, and termination
  time. Pin from the paper.
- Specific EOS implementation (Timmes Helmholtz table, or an
  equivalent). The paper's choice is the pinned target; our
  implementation may differ in algorithmic detail so long as
  convergence behavior reproduces.
- Tolerance on "converged detonation speed" — likely a small
  percentage of Chapman–Jouguet velocity at the stated
  reference resolution.
- Whether the engine-level EOS and network choices motivate a
  follow-on ADR (likely yes; no ADR presently covers
  microphysics).

## Roadmap interaction

Replicating this target requires microphysics (EOS + reaction
network) earlier in the implementation order than MHD. A
follow-up roadmap PR reorders the epochs accordingly; this plan
is a load-bearing input to that reorder.
