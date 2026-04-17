# Project Status

> **Keep this file current.** Update it as part of any PR that completes a
> milestone, starts a new epoch, or meaningfully changes what work is next.
> Maintenance and tooling PRs that do not advance the roadmap may note
> "No change to roadmap position."

---

## Current position

**Epoch 2 — design prerequisites in progress.**

Epoch 1 complete. Verification infrastructure merged. Epoch 2 design
prerequisites are being resolved before the task-graph driver is implemented.

## Completed epochs

| Epoch | Scope | Completed |
|------:|-------|-----------|
| 0 | Project scaffolding, packaging, CI, docs, ADR process, visualization scaffolding, `hello` | PR #30 area |
| 1 | Op / Region / Policy / Dispatch kernel interface; JAX `FlatPolicy`; Field placement; parallel HDF5 I/O | PR #62 (close-out sweep) |

## Completed (pre-Epoch 2)

**Verification infrastructure:**

| Item | PR | Status |
|------|----|--------|
| ADR-0007 amendment: external grounding for physics capability tests | #63 | Merged |
| Formula register (`replication/formulas.md`) | #64 | Merged |
| Convergence-order measurement helper (`tests/utils/convergence.py`) | #69 | Merged |
| SymPy stencil coefficient verification (`tests/utils/stencils.py`) | #70 | Merged |
| Epoch 2 design prerequisites: global reduction + diagnostics | #68 | Merged |

**Epoch 2 design prerequisites:**

| Item | PR | Status |
|------|----|--------|
| Field name → Dispatch input binding (`BoundOp` protocol, ADR-0010 amendment) | #74, #75 | Merged |
| Halo fill fence (`HaloFillFence` + `HaloFillPolicy`, ADR-0011) | #76 | Merged |
| Global reduction primitive for simulation diagnostics (ADR-0012) | #77 | Merged |

## Next planned work

All three Epoch 2 design prerequisites are resolved. Begin
**Epoch 2 — Mesh and AMR** per `roadmap/epoch-02-mesh.md`.

## Reference

Full epoch plan: [`roadmap/index.md`](roadmap/index.md)
Per-epoch details: `roadmap/epoch-NN-*.md`
