# Capability: aprox-family nuclear reaction network

- **ID:** C0007
- **Status:** Proposed
- **Implemented in:** TBD

## Behavior

Compact nuclear reaction network suitable for thermonuclear
burning in the Chandrasekhar-mass regime (aprox13 expected).
Given (ρ, T, Y_i), returns the energy generation rate, species
source terms, and the Jacobian required by an implicit
integrator.

## Dependents

- P01 in castro-detonation (1-D nuclear ignition convergence)

## Open questions

Populated when this capability is first tackled. The "symbolic
vs hand-written microphysics" crossroad in
`roadmap/object-level/README.md` §Crossroads / Open Decisions governs the
Jacobian-provisioning
decision.
