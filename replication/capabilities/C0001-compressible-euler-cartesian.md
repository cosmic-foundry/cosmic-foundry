# Capability: Compressible Euler solver on Cartesian grids

- **ID:** C0001
- **Status:** Proposed
- **Implemented in:** TBD

## Behavior

Finite-volume solver for the compressible Euler equations on
uniform Cartesian grids in 1-D and 2-D. Consumes an EOS through
an abstract interface (a concrete implementation is plugged in
per-problem — see C0002, C0006). Supports periodic and outflow
boundary conditions. Well-balancing is required if any
dependent problem mandates hydrostatic balance under a body
force (see C0005).

## Dependents

- P01 in castro-wd-merger (KH Galilean invariance)
- P01 in castro-detonation (1-D nuclear ignition convergence)

## Open questions

Populated when this capability is first tackled. The dependent
problem specs list the target-code properties that must be
pinned at that time, including the fiducial scheme choice and
any well-balancing requirement.
