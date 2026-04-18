# Epoch 7 — Microphysics sub-layer

> Part of the [Cosmic Foundry roadmap](index.md).

Bring up the equations of state and reaction networks that later
physics modules depend on:

- Abstract EOS interface with ideal-gas, Helmholtz, piecewise
  polytropic, and tabulated nuclear finite-T implementations.
  Tables are JAX-jittable piecewise interpolants.
- Reaction-network engine with autodiff-generated Jacobians,
  α-network reference, and a path to large adaptive networks.
- Primordial and metal cooling tables.
- Radiation opacities.

**Exit criterion:** thermonuclear flame and primordial cooling
benchmarks match published results, and a 1-D thermonuclear-flame
explainer ships with interactive EOS / network switching.

## Pre-entry checklist

Before starting Epoch 6 work, complete the following:

- **ADR-0008 decision review.** The stub for ADR-0008
  (numerical-transcription discipline) reserves the number and frames
  the problem but defers the final decision. Microphysics capabilities
  (EOS, reaction networks) will be the first major code bodies large
  enough and numeric-heavy enough to fully stress the
  question. Reread [ADR-0008](../adr/ADR-0008-numerical-transcription-discipline.md),
  review how Epochs 2–5 physics landed (especially dense numeric
  tables and formula derivations), and either finalize the decision
  via an amendment or supersede it with a full ADR. The outcome is
  not a surprise — follow the ADR-family review protocol (per `AI.md`
  → *Epoch retrospective* → *ADR set as a whole*) to surface any
  overlaps or reframings with adjacent ADRs (0005, 0007) before
  finalizing.
