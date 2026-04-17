# Project Status

> **Keep this file current.** Update it as part of any PR that completes a
> milestone, starts a new epoch, or meaningfully changes what work is next.
> Maintenance and tooling PRs that do not advance the roadmap may note
> "No change to roadmap position."

---

## Current position

**Between Epoch 1 and Epoch 2.**

Epoch 1 (kernel abstraction layer + infrastructure) is complete. Verification
infrastructure is being added before Epoch 2 begins. Epoch 2 has not started.

## Completed epochs

| Epoch | Scope | Completed |
|------:|-------|-----------|
| 0 | Project scaffolding, packaging, CI, docs, ADR process, visualization scaffolding, `hello` | PR #30 area |
| 1 | Op / Region / Policy / Dispatch kernel interface; JAX `FlatPolicy`; Field placement; parallel HDF5 I/O | PR #62 (close-out sweep) |

## In progress

**Verification infrastructure** (no epoch number; these are cross-cutting
tools that all subsequent epochs depend on):

| Item | PR | Status |
|------|----|--------|
| ADR-0007 amendment: external grounding for physics capability tests | #63 | Merged |
| Formula register (`replication/formulas.md`) | #64 | Merged |
| Convergence-order measurement helper (`tests/utils/convergence.py`) | #69 | Merged |
| SymPy stencil coefficient verification (`tests/utils/stencils.py`) | #70 | Merged |
| Epoch 2 design prerequisites: global reduction + diagnostics | #68 | Merged |

## Next planned work

1. Begin **Epoch 2 — Mesh and AMR** per `roadmap/epoch-02-mesh.md`.
   Design prerequisites (read that file before starting):
   - Field name → Dispatch input binding protocol
   - Halo fill fence mechanism
   - Global reduction primitive for simulation diagnostics

## Reference

Full epoch plan: [`roadmap/index.md`](roadmap/index.md)
Per-epoch details: `roadmap/epoch-NN-*.md`
