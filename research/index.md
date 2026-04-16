# Computational Astrophysics Code Landscape — Research Notes

This document surveys major computational astrophysics codes that are
relevant to the Cosmic Foundry engine design. For each code it records
canonical papers, public source location, license, and capabilities.
The survey culminates in a **union of capabilities** that this
repository should ultimately be able to cover if the goal is parity
(or superset) with the physics that any of these codes can simulate.

**Scope note.** Cosmic Foundry is intended to be a fully self-contained
engine. The frameworks surveyed below (AMReX, Kokkos, Parthenon,
Charm++, Chombo, Cactus, MPI-AMRVAC) are therefore studied as
**capability references whose functionality must be replicated inside
this repository**, not as dependencies to be linked against. Their
papers are cited because they document the algorithms we would need
to re-implement — block-structured AMR, performance-portable parallel
loops, task-based asynchronous drivers, multigrid solvers, particle
infrastructure, and so on. Where a feature of a downstream physics
code is attributed to "AMReX" or "Parthenon," read that as shorthand
for *capabilities the engine must provide natively*.

The survey is organized by infrastructure family rather than strictly
alphabetically, because most modern codes cluster around a small set
of these framework design patterns.

## Contents

| § | File | Scope |
|---|------|-------|
| 1 | [01-frameworks.md](01-frameworks.md) | AMReX, Kokkos, Parthenon, Charm++, supporting libraries, and GPU kernel DSLs (Singe, PyJac, TChem, Legion/Regent). |
| 2 | [02-grid-codes.md](02-grid-codes.md) | Structured-grid / AMR finite-volume codes (Castro, Nyx, FLASH-X, GAMER, Athena++, AthenaK, Enzo, RAMSES, PLUTO, MPI-AMRVAC, ZEUS, Dedalus, …). |
| 3 | [03-particle-codes.md](03-particle-codes.md) | Particle-based and meshless codes (Arepo, GADGET-4, GIZMO, ChaNGa, SWIFT). |
| 4 | [04-relativistic-codes.md](04-relativistic-codes.md) | Relativistic / GRMHD / numerical-relativity codes (Phoebus, Einstein Toolkit, SpECTRE, BHAC, KORAL, HARM, KHARMA, GRChombo, Spritz, WhiskyTHC). |
| 5 | [05-stellar-codes.md](05-stellar-codes.md) | Stellar structure / evolution and CCSN microphysics (MESA, KEPLER, CHIMERA, FORNAX, thornado, WeakLib). |
| 6 | [06-capabilities.md](06-capabilities.md) | §6.1–§6.10 — the union of capabilities that a self-contained engine must cover. |
| 6.11 | [06-11-visualization.md](06-11-visualization.md) | §6.11 — visualization and science-communication landscape (broken out because the section is long and evolves faster than the code survey). |
| 6.12 | [06-12-licensing.md](06-12-licensing.md) | §6.12 — licensing and openness landscape for the surveyed codes. |
| 7 | [07-implications.md](07-implications.md) | §7 — implications for Cosmic Foundry. |

## Splitting discipline

The survey was split out of a single 1000+ line `RESEARCH.md` in 2026
so each section can grow independently without inflating the rest.
Further per-code splitting (one file per code) is an option if any one
section grows past ~300 lines, but is not adopted yet — the current
per-section granularity matches the document's outline and keeps
cross-references stable (§N.M numbers are preserved).
