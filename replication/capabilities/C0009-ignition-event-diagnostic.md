# Capability: Ignition-event diagnostic

- **ID:** C0009
- **Status:** Proposed
- **Implemented in:** TBD

## Behavior

Detects and records the first self-sustained thermonuclear
ignition event in a simulation: its time t_ign and spatial
location x_ign. The detection criterion — typically based on
local energy-generation rate vs. loss, a species threshold, or
a combination — is load-bearing for the detonation problem's
success criterion and must be pinned from the paper.

## Dependents

- P01 in castro-detonation (1-D nuclear ignition convergence)

## Open questions

Populated when this capability is first tackled. Sub-timestep
and sub-cell localization of t_ign and x_ign are expected to
live here, but could be deferred to a post-processing step in
the test harness.
