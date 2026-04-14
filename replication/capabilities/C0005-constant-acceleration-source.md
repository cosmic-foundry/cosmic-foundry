# Capability: Constant acceleration source term

- **ID:** C0005
- **Status:** Proposed
- **Implemented in:** TBD

## Behavior

A fixed uniform body force added to the momentum equation, with
its associated work term added to the energy equation. Used as
a stand-in for gravity in problems that do not need a self-
gravity solve. Well-balanced operation — preservation of a
hydrostatic-balanced initial pressure profile to roundoff under
zero-velocity evolution — is required when a dependent problem
mandates it.

## Dependents

- P01 in castro-detonation (1-D nuclear ignition convergence)

## Open questions

Populated when this capability is first tackled. Whether the
well-balanced property is required unconditionally, or only
when a dependent problem demands hydrostatic balance, is pinned
by the detonation problem.
