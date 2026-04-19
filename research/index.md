# Computational Astrophysics Code Landscape — Research Notes

This document surveys major computational astrophysics codes. For each
code it records canonical papers, public source location, license, and
capabilities. The survey is organized by infrastructure family rather
than alphabetically, because most modern codes cluster around a small
set of framework design patterns.

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
| 8 | [08-vv-methodology.md](08-vv-methodology.md) | §8 — V&V and specification methodology: Roache, Oberkampf & Roy, MMS, PCMM, and what exists (and doesn't) for specification-first code development. |

## Splitting discipline

The survey was split out of a single 1000+ line `RESEARCH.md` in 2026
so each section can grow independently without inflating the rest.
Further per-code splitting (one file per code) is an option if any one
section grows past ~300 lines, but is not adopted yet — the current
per-section granularity matches the document's outline and keeps
cross-references stable (§N.M numbers are preserved).
