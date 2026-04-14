# Capability: Linear-mode growth-rate diagnostic

- **ID:** C0004
- **Status:** Proposed
- **Implemented in:** TBD

## Behavior

Given a field time series and a known mode (wavenumber,
direction), extract the Fourier amplitude of that mode per
snapshot and fit an exponential over the linear-growth phase.
Returns the fit growth rate and its uncertainty. Must itself be
Galilean-invariant, or its boost correction must be part of the
contract.

## Dependents

- P01 in castro-wd-merger (KH Galilean invariance)

## Open questions

Populated when this capability is first tackled.
