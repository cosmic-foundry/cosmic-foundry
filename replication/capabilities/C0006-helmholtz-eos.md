# Capability: Helmholtz-family degenerate stellar EOS

- **ID:** C0006
- **Status:** Proposed
- **Implemented in:** TBD

## Behavior

Tabulated EOS in the Helmholtz family, covering the regime
relevant to degenerate stellar matter. Supplies pressure,
internal energy, sound speed, and the thermodynamic derivatives
required both by C0001 and by the reaction-network integrator
(C0007, via C0008).

## Dependents

- P01 in castro-detonation (1-D nuclear ignition convergence)

## Open questions

Populated when this capability is first tackled. Whether an
abstract EOS interface unifying C0002 and C0006 is worth
introducing now, or left until a third EOS capability lands, is
a design choice surfaced here.
