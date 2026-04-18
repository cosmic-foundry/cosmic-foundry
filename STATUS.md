# Project Status

> **Keep this file current.** Update it as part of any PR that completes a
> milestone, starts a new epoch, or meaningfully changes what work is next.
> Maintenance and tooling PRs that do not advance the roadmap may note
> "No change to roadmap position."

---

## Current position

**Object-level track:** Epoch 2 — Mesh and AMR — in progress.

All design prerequisites resolved. Implementation underway per the
sequencing plan in `roadmap/epoch-02-mesh.md`.

**Meta-level track:** Reproducibility meta-generator M3 planned.
Implement the platform-only capsule convergence slice in
`roadmap/reproducibility-meta-generator.md`.

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

## Epoch 2 progress

Per the implementation plan in `roadmap/epoch-02-mesh.md`:

| # | Item | PR | Status |
|---|------|----|--------|
| 1 | Uniform mesh data model (`Block`, `UniformGrid`) | #85 | Merged |
| 2 | Field allocation from blocks | #88 | Merged |
| 3 | `HaloFillPolicy` — single-rank | #90 | Merged |
| 4 | `DiagnosticReducer` + `DiagnosticSink` | #92 | Merged |
| 5 | Task-graph driver — single-rank | — | Planned |
| 6 | `HaloFillPolicy` — multi-rank | — | Planned |

Items 7–13 (AMR hierarchy, I/O, exit criterion) in
`roadmap/epoch-02-mesh.md`.

## Meta-level progress

Per the two-track roadmap in `roadmap/index.md`:

| ID | Item | PR | Status |
|----|------|----|--------|
| M0 | Branch / PR / attribution discipline | #8 area | Merged |
| M1 | Replication workflow and externally grounded tests | #63, #64, #69, #70 | Merged |
| M2 | Derivation-first lane | #81 area | Merged |
| M3 | Reproducibility meta-generator architecture and convergence plan | #93 | Planned |
| M4 | Platform validation manifests, provenance, comparison-result schema, sim-spec format | — | Planned |

## Next planned work

Next selected PRs should explicitly state which track they advance.

Meta-level next work: implement
`roadmap/reproducibility-meta-generator.md` M3 — platform-only capsule
collect, dry-run, render, and structural comparison against the current
repository state.

Object-level next work: continue Epoch 2 items #5–#6 (task-graph driver,
multi-rank halo fill), then begin Epoch 3 (Platform Services): manifest
infrastructure, comparison-result schema, and simulation specification
format. See `roadmap/epoch-03-platform-services.md` for the full plan.

## Reference

Full epoch plan: [`roadmap/index.md`](roadmap/index.md)
Per-epoch details: `roadmap/epoch-NN-*.md`
Meta-generator roadmap: `roadmap/reproducibility-meta-generator.md`
