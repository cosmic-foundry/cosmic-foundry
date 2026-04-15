# 7. Implications for Cosmic Foundry

> Part of the [Cosmic Foundry research notes](index.md).
>
> **Note.** The source `RESEARCH.md` contained two distinct `## 7.
> Implications` sections (lines 1045 and 1150 in the pre-split
> document). Both are preserved verbatim below, as the mechanical
> split is lossless; consolidating or removing the duplication is
> an editorial question tracked separately.

## 7a. Implications — infrastructure / physics decomposition

The goal is a **fully self-contained engine** — so AMReX, Parthenon,
Kokkos, Chombo, Cactus, Charm++, and MPI-AMRVAC are *capability
targets to replicate*, not dependencies to adopt. This shifts the
design task from "pick a substrate and plug physics packages in" to
"identify the minimal set of infrastructure primitives we must build
before any physics package can stand on them." The union of
capabilities in §6 collapses, at the engine level, into roughly
these foundations:

1. **Core infrastructure layer (must be implemented from scratch).**
   - A block-structured AMR data model with cell/face/edge/node
     centering, subcycling in time, and refinement-flux correction
     (functionally equivalent to what AMReX and Parthenon provide).
   - A performance-portability abstraction over CPU SIMD, CUDA, HIP,
     and SYCL execution — conceptually a Kokkos analog — so that
     kernels written once compile and run across vendor GPUs and
     CPUs. The research survey (§1.2 Kokkos, §1.6 Singe) reveals
     that this abstraction must keep three axes independent: an *Op*
     (per-element callable with declared stencil footprint, analogous
     to a Kokkos `__device__` function), a *Region* (spatial
     sub-domain over which an Op is applied, potentially a packed
     collection of meshblocks analogous to Parthenon's
     `MeshBlockPack`), and a *Policy* (execution organization —
     flat, tiled-with-scratchpad, or warp-specialized — controlling
     how threads are arranged to process a Region). The dispatch
     unit, a *Pass*, composes these three. Kernel-launch granularity
     is controlled at the Pass/Policy level; Ops are unaware of it.
     This separation allows both fusion experiments and execution
     policy substitution without touching physics code. See ADR-0010.
   - A task-based asynchronous driver with explicit dependency
     graphs, over-decomposition, and dynamic load balancing (the
     role played elsewhere by Athena++'s task list, Parthenon's
     driver, SWIFT's task graph, and Charm++). Note: this is
     *separate* from the Pass/Region batching above. The task graph
     controls sequencing and communication overlap between logical
     work units; Region batching controls how many items of work
     enter one kernel launch. Parthenon implements these separately
     as `TaskList` and `MeshBlockPack`; the same separation applies
     here.
   - A particle / swarm infrastructure with particle–mesh
     operators, tree and FMM gravity, and adaptive softening.
   - Linear solver suite: geometric multigrid (cell-centered and
     nodal), FFT-based Poisson, and interfaces for iterative /
     Krylov solvers on the AMR hierarchy.
   - Parallel I/O: a native checkpoint/plotfile format plus HDF5
     export; VTK / yt-compatible output.

2. **Mesh paradigms beyond block AMR.** Grid codes alone cannot
   cover every niche, so the engine should also provide:
   - An SPH / meshless finite-mass or finite-volume path (GIZMO,
     SWIFT, GADGET-4 territory).
   - A Voronoi / moving-mesh path (Arepo territory) — the hardest
     to replicate, but the only way to match Galilean-invariant
     galaxy-formation work.
   - Optional spectral bases for smooth problems (Dedalus
     territory).

3. **Pluggable physics packages** layered on the infrastructure,
   each independently selectable: Newtonian hydro/MHD, non-ideal
   MHD, SR and GR (M)HD, dynamical-spacetime NR (BSSN/Z4c/CCZ4),
   radiation (FLD, M1, short-characteristics, DG M1 for neutrinos,
   Monte Carlo), chemistry/cooling, cosmic rays, dust.

4. **A microphysics sub-layer** — ideal / Helmholtz / piecewise-
   polytropic / tabulated nuclear EOS, reaction networks from
   α-networks through large adaptive networks, radiation opacities,
   neutrino interaction sets, primordial and metal cooling tables.
   These can be implemented in-repo using the algorithmic
   descriptions in the papers of AMReX-Astro Microphysics,
   Singularity-EOS/Opac, WeakLib, and Grackle; the engine must
   own the code.

5. **Gravity / N-body / cosmology stack** — TreePM, FMM, PM, comoving
   integrator with expansion factor, 2LPT initial conditions, FOF
   and SUBFIND-style halo finders, on-the-fly light-cones and power
   spectra, δf massive neutrinos.

6. **Subgrid physics recipes** — cooling, star formation, stellar
   and SN feedback, AGN / BH seeding and feedback, chemical
   enrichment — exposed as a plugin interface so EAGLE-, COLIBRE-,
   and FIRE-style recipes can be expressed within the engine.

7. **Stellar evolution module** — 1-D Lagrangian structure with
   implicit solver, adaptive mesh and timestep, coupled nuclear
   burning, rotation, mixing, diffusion, and binary evolution
   (MESA / KEPLER territory) — kept compatible with the multi-D
   explosive codes so that progenitor states flow naturally into
   supernova / merger runs.

8. **Solver / time-stepping toolkit** — explicit, implicit, and
   IMEX integrators; spectral deferred corrections; super-time-
   stepping for parabolic terms; Anderson-accelerated nonlinear
   solvers; primitive-variable recovery for GRMHD with multiple
   robust fallbacks.

9. **Diagnostics and observables** — on-the-fly synthetic EUV /
   X-ray / spectral-line output, halo / merger-tree builders,
   integrated analysis hooks, in-situ visualization.

**Strategic implication.** The realistic ordering of work is
*infrastructure first, physics second*. A Cosmic Foundry built
this way would look, structurally, a lot like Parthenon or AMReX
at its lowest layer, with an Athena++/AthenaK-like task-driven
physics layer on top. Because most of the permissive-licensed
reference codes (Castro, Flash-X, AthenaK, Parthenon, KHARMA,
Phoebus, thornado, Enzo-E) are freely readable, clean-room
re-implementation is practical: the papers and source can be
studied directly. Copyleft codes (Arepo, RAMSES, GADGET-4, MESA,
SWIFT, PLUTO) should be consulted through their published
algorithm descriptions only, with reimplementation done from those
descriptions rather than by reading source.

No single existing code covers the full union of §6; the closest
published superset would be "Flash-X + AthenaK + SWIFT + Arepo +
MESA." Reproducing that union inside one self-contained repository
is a long-horizon program, but it is tractable if the infrastructure
layer is built well enough that each physics module becomes a
roughly paper-scale effort rather than a full-code-scale one.

---

## 7b. Implications — ten-bullet summary

A single engine aspiring to the union of the capabilities above would
need, at minimum:

1. **A performance-portable AMR substrate.** The strongest current
   precedents are AMReX (Castro/Quokka/Flash-X) and Kokkos+Parthenon
   (Phoebus/KHARMA/AthenaK/AthenaPK). Both are BSD-3-Clause and
   interoperable with MPI, HDF5, and modern GPUs.
2. **A pluggable physics "package" architecture** in the style of
   Parthenon or Enzo-E, so that hydro, MHD, GR, radiation, chemistry,
   gravity, and particles can be composed per problem without
   recompiling the world.
3. **Multiple mesh paradigms** — at least block AMR, particle/SPH,
   and (aspirationally) moving-mesh — because Arepo-style moving
   meshes and particle/meshless methods occupy real niches (galaxy
   formation, merging stars) that grid codes do not naturally cover.
4. **Relativistic physics as first-class**, not an afterthought:
   GR metric evolution (BSSN/Z4c/CCZ4), GRMHD with CT, primitive
   recovery, and M1/MC neutrino transport.
5. **A microphysics layer** covering ideal/Helmholtz/tabulated EOS,
   large reaction networks, primordial + metal chemistry, neutrino
   opacities, radiation opacities — reusing existing open libraries
   (AMReX-Astro Microphysics, Singularity-EOS / Singularity-Opac,
   WeakLib, Grackle) rather than reinventing them.
6. **Subgrid / galaxy-formation "physics recipes"** — cooling, SF,
   stellar and AGN feedback, chemical enrichment — with a plugin
   interface so community models (EAGLE/COLIBRE/FIRE-equivalents)
   can be swapped.
7. **Cosmological and N-body machinery** — comoving integrator,
   TreePM / FMM gravity, 2LPT ICs, FOF/SUBFIND, light-cones, power
   spectra.
8. **A stellar-evolution module** covering long-timescale 1-D
   Lagrangian evolution with adaptive mesh + implicit solvers, so
   progenitor states can flow directly into multi-D explosive codes.
9. **Solver infrastructure** — IMEX, SDC, super-time-stepping,
   Anderson-accelerated nonlinear solvers, multigrid, FFT, FMM.
10. **Permissive licensing** (BSD/Apache/MIT) to allow borrowing
    algorithms and coupling to downstream commercial or
    government-use work without copyleft entanglements.

No single existing code covers all ten bullets; the closest superset
today would be "Flash-X + AthenaK + SWIFT + Arepo + MESA", unified
under an AMReX-or-Parthenon substrate. Cosmic Foundry's practical
trajectory is therefore to pick one substrate (AMReX or Parthenon),
reuse open microphysics libraries, and add the particle/moving-mesh
and stellar-evolution pieces that neither substrate currently
provides natively.
