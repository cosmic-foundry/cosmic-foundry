# 6.12 Licensing and openness landscape

> Part of the [Cosmic Foundry research notes](index.md).

Because Cosmic Foundry aims to be self-contained — re-implementing
rather than linking to these codes — licenses matter as a constraint
on **which reference implementations we can study closely and adapt
algorithms from** without inheriting copyleft obligations on our own
source.

- **Permissive (BSD / Apache / MIT) — safe to study and adapt freely:**
  Castro, Nyx, MAESTROeX, Quokka, GAMER, Athena++, AthenaK, KHARMA,
  Parthenon, AMReX, Kokkos, Phoebus, GRChombo, thornado, Flash-X,
  Enzo/Enzo-E. Algorithms from these codes can be reproduced with
  attribution and without forcing the rest of the repository into a
  copyleft license.
- **Copyleft (GPL / LGPL / CeCILL) — read for design; do not copy
  code verbatim:** Arepo, GADGET-4, GIZMO-public, ChaNGa, RAMSES,
  MPI-AMRVAC, BHAC, koral_lite, PLUTO, Dedalus, MESA (LGPL), SWIFT
  (LGPL). Papers and documentation describe the algorithms freely;
  those descriptions are what we work from, and any reimplementation
  must be clean-room.
- **Closed / collaboration-only:** KEPLER, CHIMERA, FORNAX,
  WhiskyTHC, and the private extensions of Arepo (IllustrisTNG,
  Auriga) and GIZMO (FIRE, STARFORGE). Only published papers are
  available; any capability parity has to be inferred from those
  descriptions plus whatever open analogues exist (e.g. EAGLE /
  COLIBRE subgrid stacks in SWIFT as an open alternative to FIRE).
