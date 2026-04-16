# 6. Union of Capabilities

> Part of the [Cosmic Foundry research notes](index.md).
>
> §6.11 (visualization and science communication) lives in
> [06-11-visualization.md](06-11-visualization.md);
> §6.12 (licensing and openness landscape) lives in
> [06-12-licensing.md](06-12-licensing.md). §-numbers are preserved
> so cross-references from the roadmap and ADRs remain stable.

The following list enumerates every capability that appears in at
least one of the codes above. A simulation engine aiming to be able
to perform the tasks of any of these codes would need, in aggregate,
the following:

## 6.1 Discretizations and meshes

- Uniform / static curvilinear grids (Cartesian, cylindrical, spherical,
  polar, shearing-box, warped / mapped).
- Structured-patch AMR (Berger–Colella style, as in Enzo, FLASH-legacy,
  Castro/Nyx/Quokka via AMReX).
- Oct-tree block-structured AMR (Athena++, AthenaK, RAMSES, GAMER,
  Enzo-E, Parthenon-based codes, Arepo/Gadget-style tree with grid).
- Fully-threaded octree with cell-level refinement (RAMSES).
- Forest-of-octrees AMR under asynchronous runtime (Enzo-E / Cello).
- Moving Voronoi mesh (Arepo).
- Meshless finite-mass / finite-volume (GIZMO MFM/MFV).
- Smoothed-particle hydrodynamics variants (GADGET-4, SWIFT SPHENIX
  and friends, GIZMO, ChaNGa).
- Embedded-boundary / cut-cell geometry (AMReX EB).
- Discontinuous Galerkin with DG–FD hybrid shock capturing (SpECTRE,
  thornado).
- Spectral bases (Chebyshev/Fourier/Jacobi/disk/ball/shell) with tau
  method for BCs (Dedalus).
- Curvilinear adaptive grids for relativistic problems (Kerr-Schild,
  MKS, FMKS, general stationary-spacetime coordinates).

## 6.2 Hyperbolic / fluid physics

- Newtonian compressible hydrodynamics: unsplit PPM, CTU, MUSCL,
  ZEUS-style staggered-mesh, high-order WENO/MP5.
- Low-Mach / pseudo-incompressible stratified hydrodynamics with
  projection-based constraint enforcement (MAESTROeX).
- Ideal MHD with constrained transport, unsplit CT, flux-CT, vector-
  potential / A-field formulations.
- Non-ideal MHD (resistive, Hall, ambipolar, Ohmic, anisotropic thermal
  conduction and viscosity).
- Two-fluid ion–neutral plasma; multi-fluid dust coupling.
- Hybrid particle-in-cell / MHD cosmic-ray coupling.
- Special-relativistic hydrodynamics and MHD.
- General-relativistic hydrodynamics and MHD on fixed stationary
  spacetimes (Kerr, Kerr-Schild, etc.).
- General-relativistic (radiation) MHD on dynamical spacetimes (Athena++
  GR, AthenaK, SpECTRE, Einstein Toolkit + Spritz/WhiskyTHC).
- Full spacetime evolution: BSSN (moving puncture), Z4c, CCZ4,
  Generalized Harmonic; Numerical-relativity constraint handling.

## 6.3 Radiation, transport, and chemistry

- Flux-limited diffusion (gray and multigroup) radiation hydrodynamics
  (Castro, FLASH-X, ZEUS-MP, Enzo implicit FLD).
- Two-moment M1 radiation hydrodynamics (Quokka, RAMSES-RT, AthenaK,
  KORAL, Fornax, thornado).
- Short-characteristics / implicit radiation transport (Athena++).
- Ray-tracing radiative transfer (Enzo+Moray).
- Monte Carlo neutrino transport (Phoebus).
- Spectral DG neutrino transport with IMEX (thornado, planned SpECTRE).
- Non-equilibrium primordial chemistry (H / He, 6/9/12-species).
- Non-equilibrium metal chemistry (RAMSES-RTZ, Grackle, CHIMES).
- Metal-line cooling, UV background, equilibrium and tabulated cooling
  (Grackle interfaces).
- Cosmic-ray transport: anisotropic diffusion, streaming, Alfvén
  coupling (GAMER, GIZMO).
- Two-temperature (ion + electron) plasma thermodynamics; non-thermal
  electron evolution (KORAL, KHARMA).

## 6.4 Gravity and cosmology

- Point-mass gravity and rotating frames.
- Monopole / multipole gravity for nearly spherical stellar problems.
- Geometric multigrid Poisson on AMR (AMReX, Parthenon, Athena++,
  Enzo-E, MPI-AMRVAC).
- FFT-based and hybrid FFT + multigrid gravity (Enzo, Athena++).
- Particle-mesh / PPPM gravity for N-body (Nyx, Enzo).
- Tree gravity (Barnes–Hut) with Ewald summation for periodicity
  (ChaNGa, GIZMO, Gadget family).
- Fast Multipole Method (GADGET-4, SWIFT FMM–PM).
- Adaptive softening.
- Comoving FRW cosmological integrator and periodic / shearing /
  vacuum boundaries.
- 2LPT / Zel'dovich initial conditions; MUSIC / NgenIC-style IC
  generation.
- Massive neutrinos via δf particle method (SWIFT).
- On-the-fly FOF and SUBFIND-HBT halo finders; merger-tree builder;
  light-cone output; high-dynamic-range power-spectrum estimator.

## 6.5 Equations of state and microphysics

- Ideal gas, polytropic, piecewise polytropic EOS.
- Helmholtz EOS (degenerate stellar).
- General stellar and tabulated nuclear finite-T EOS (SFHo, LS220, DD2,
  Lattimer–Swesty).
- Singularity-EOS / WeakLib / EOS_Omni interfaces for tabulated
  high-energy-density EOS.
- Nuclear reaction networks from α-networks through ~ 2000 isotopes
  (Aprox13/19, XNet, adaptive BURN, AMReX-Astro Microphysics).
- Neutrino–matter interaction sets: Bruenn baseline + weak magnetism,
  many-body, bremsstrahlung, electron capture on heavies, inelastic
  scattering.
- Dust dynamics and grain chemistry hooks.
- Laser drive and multi-temperature HED plasma (Flash-X HEDP).

## 6.6 Subgrid / astrophysical "physics modules"

- Schmidt-law star formation, Springel–Hernquist effective ISM.
- Stellar and SNe feedback (thermal, kinetic, delayed-cooling).
- AGN feedback via sink / black-hole particles with seeding.
- Chemical enrichment tracking (metals, isotopes).
- Pop III / Pop II formation and feedback recipes (Enzo).
- Fuzzy / wave dark matter (ELBDM) — GAMER-2 distinguishing feature.
- Cosmic-ray injection and propagation.
- Sink particles for star-forming clumps / binaries (STARFORGE).
- Tracer particles, passive scalars, active particles (PIC, dust).
- Immersed-boundary and fluid–structure interaction (Flash-X).
- Neutrino leakage and moment transport for CCSN / BNS mergers.

## 6.7 Stellar evolution

- 1-D Lagrangian stellar structure + evolution with implicit solver,
  adaptive mesh and timestep (MESA, KEPLER).
- Full rotation (shellular), magnetic braking, Tayler–Spruit dynamo.
- Element diffusion, gravitational settling.
- Time-dependent convection, semiconvection, overshooting,
  thermohaline mixing; boundary-layer treatments.
- Binary evolution: mass transfer, simplified CE, tidal synchronization.
- Explicit stellar-explosion and pulsation capability; asteroseismic
  diagnostics (GYRE interface).
- Coupled nucleosynthesis with very large networks.

## 6.8 Numerics / solver technology

- Implicit, explicit, and IMEX time integrators (including Anderson-
  accelerated solvers for radiation–matter coupling).
- Spectral deferred corrections (SDC) high-order time stepping
  (Castro).
- Super-time-stepping for parabolic terms (Athena++, MPI-AMRVAC).
- Subcycling in time across AMR levels.
- Task-based asynchronous drivers with dependency graphs
  (SWIFT, Parthenon, Athena++/AthenaK, SpECTRE).
- Over-decomposed message-driven runtime (Charm++: ChaNGa, Enzo-E,
  SpECTRE).
- Adaptive softening, hierarchical time-stepping (Gadget-family).
- Multiple primitive-variable recovery schemes for GRMHD (2D/1D Noble,
  Palenzuela-Newman, RePrimAnd, entropy fallback).
- Bitwise-reproducible runs (GAMER-2, GADGET-4).

## 6.9 Parallelism and performance portability

- MPI distributed memory; OpenMP / pthreads / MPI-3 shared windows;
  SIMD intrinsics.
- Single-source GPU execution across CUDA, HIP, SYCL (AMReX, Kokkos,
  and downstream codes).
- GPU tree-walks (ChaNGa) and GPU-resident AMR (Parthenon, AthenaK,
  Phoebus, KHARMA, Quokka).
- Async one-sided MPI communication; on-the-fly load balancing
  (Hilbert SFC, ORB, Charm++ balancers).
- Dynamic mesh/work redistribution with fault-tolerance hooks
  (Charm++).

"Performance portability" in the codes above covers two distinct
programming models, and the distinction matters for how physics kernels
are authored:

**Hierarchical model (Kokkos + downstream: Parthenon, AthenaK, KHARMA,
Phoebus, AthenaPK).** The programming model exposes three execution
levels — a grid of thread teams, threads within each team, and vector
lanes within each thread — plus a corresponding scratchpad memory
hierarchy (HBM → team SRAM → per-thread registers). The physics author
chooses the execution policy, declares scratchpad size, issues explicit
barriers, and writes cooperative load / compute phases. The framework
compiles each concept to the hardware primitive: team → CUDA block /
OpenMP region; team scratchpad → `__shared__` / stack; barrier →
`__syncthreads()` / no-op. GPU performance for bandwidth-bound stencil
operations (reconstruction, divergence, Laplacian) depends on correct
use of this hierarchy by the physics author.

**Flat model (AMReX + downstream: Castro, Quokka, Flash-X, WarpX,
MAESTROeX).** The programming model exposes only flat per-element
iteration (`ParallelFor`). Physics authors write per-cell lambdas with
no awareness of thread teams, scratchpad, or barriers. GPU performance
on memory-hierarchy-sensitive paths comes from AMReX's own pre-tuned
framework kernels and from vendor library calls (cuFFT, cuSolver,
Hypre), not from physics-author-controlled tiling. The tradeoff is
a simpler authoring model at the cost of less direct control over
the memory hierarchy in user-written physics code.

## 6.10 I/O, diagnostics, and ecosystem

- Parallel HDF5 and PnetCDF; AMReX plotfiles; ADIOS2; VTK/yt-
  compatible output.
- On-the-fly synthetic observables (EUV, X-ray, spectra).
- Integrated analysis hooks (yt, VisIt, ParaView, Ascent).
- On-the-fly halo-finding, merger trees, light-cones, power spectra.
