# Computational Astrophysics Code Landscape — Research Notes

This document surveys major computational astrophysics codes that are
relevant to the Cosmic Foundry engine design. For each code it records
canonical papers, public source location, license, and capabilities.
A final section synthesizes the **union of capabilities** that this
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

---

## 1. Frameworks & Infrastructure Libraries

### 1.1 AMReX

Block-structured adaptive mesh refinement framework (LBNL / ECP).

- **Papers:**
  - Zhang et al. (2019), *AMReX: a framework for block-structured AMR*,
    JOSS 4(37), 1370. DOI: [10.21105/joss.01370](https://doi.org/10.21105/joss.01370)
  - Zhang et al. (2021), *AMReX: Block-Structured AMR for Multiphysics
    Applications*, IJHPCA 35, 508. arXiv:
    [2009.12009](https://arxiv.org/abs/2009.12009)
- **Source:** <https://github.com/AMReX-Codes/amrex>
- **License:** BSD-3-Clause
- **Capabilities:** block-structured AMR hierarchy; cell/face/edge/node
  data; embedded-boundary cut-cell geometry; subcycling in time;
  geometric multigrid linear solvers (cell-centered, nodal, tensor);
  optional Hypre/PETSc back-ends; particle data structures (AoS/SoA)
  with particle–mesh operators; native plotfile + HDF5 + ADIOS2 I/O;
  performance portability across MPI, OpenMP, CUDA, HIP, SYCL.
- **Downstream codes:** Castro, Nyx, MAESTROeX, Quokka, WarpX, PeleC,
  IAMR, Flash-X (AMR driver), CarpetX (Einstein Toolkit).

### 1.2 Kokkos

C++ performance-portability model (Sandia / ECP).

- **Papers:**
  - Edwards, Trott, Sunderland (2014), *Kokkos: Enabling manycore
    performance portability...*, JPDC 74, 3202. DOI:
    [10.1016/j.jpdc.2014.07.003](https://doi.org/10.1016/j.jpdc.2014.07.003)
  - Trott et al. (2021), *The Kokkos EcoSystem*, CiSE 23(5), 10.
    DOI: [10.1109/MCSE.2021.3098509](https://doi.org/10.1109/MCSE.2021.3098509)
  - Trott et al. (2022), *Kokkos 3: Programming Model Extensions for
    the Exascale Era*, IEEE TPDS.
- **Source:** <https://github.com/kokkos/kokkos>
- **License:** Apache-2.0 WITH LLVM-exception (BSD-3-Clause pre-4.0)
- **Capabilities:** single-source execution on CUDA, HIP, SYCL, OpenMP,
  OpenMPTarget, HPX, C++ threads, Serial; `View` memory abstractions;
  hierarchical parallelism; Kokkos Kernels (BLAS/sparse), Tools, FFT,
  Remote Spaces.
- **Downstream codes:** Parthenon, AthenaK, K-Athena (archived),
  KHARMA, Phoebus, AthenaPK.

### 1.3 Parthenon

Performance-portable block-structured AMR framework derived from
Athena++ / K-Athena (LANL / MSU / ORNL).

- **Paper:** Grete et al. (2023), *Parthenon — a performance portable
  block-structured AMR framework*, IJHPCA 37(5), 465. arXiv:
  [2202.12309](https://arxiv.org/abs/2202.12309)
- **Source:** <https://github.com/parthenon-hpc-lab/parthenon>
- **License:** BSD-3-Clause
- **Capabilities:** octree block-based AMR; Kokkos-backed
  device-resident data; task-based driver; variable/meshblock packing
  to amortize kernel launches; cell/face/edge/node fields; particle
  swarms; asynchronous one-sided MPI; plugin "Packages". Scales to
  ~1.7 × 10¹³ zone-cycles/s on 9,216 Frontier nodes.
- **Downstream codes:** AthenaPK, KHARMA, Phoebus, Riot.

### 1.4 Charm++

Message-driven, over-decomposed asynchronous runtime used by ChaNGa,
Enzo-E, and SpECTRE. (Not a simulation library itself, but the
parallelism model of three relevant codes.) Source:
<https://github.com/charmplusplus/charm>, license: University of
Illinois / NCSA Open Source License.

### 1.5 Other supporting libraries

- **Chombo** (LBNL) — Berger–Rigoutsos AMR used by PLUTO (AMR build)
  and GRChombo; public, BSD-style. <https://github.com/applied-numerical-algorithms-group-lbnl/Chombo>
- **Cactus** — modular "flesh+thorns" framework underlying the
  Einstein Toolkit; LGPL. <https://cactuscode.org/>
- **MPI-AMRVAC** — AMR framework that hosts BHAC. <https://github.com/amrvac/amrvac>
- **Grackle** — chemistry/cooling library (primordial + metal line
  cooling, UV background); BSD-3-Clause.
  <https://github.com/grackle-project/grackle>
- **Singularity-EOS / Singularity-Opac** — LANL EOS and opacity
  libraries used by Phoebus.
- **WeakLib** — tabulated nuclear EOS + neutrino opacities used by
  thornado / Flash-X CCSN applications.

---

## 2. Structured-Grid / AMR Finite-Volume Codes

### 2.1 Castro

Compressible multi-physics astrophysics on AMReX.

- **Papers:**
  - Almgren et al. (2010), *CASTRO I: Hydrodynamics and Self-gravity*,
    ApJ 715, 1221. arXiv: [1005.0114](https://arxiv.org/abs/1005.0114)
  - Zhang et al. (2011), Castro radiation hydrodynamics, ApJS 196, 20.
    arXiv: [1105.2466](https://arxiv.org/abs/1105.2466)
  - Zhang et al. (2013), Multigroup radiation hydrodynamics, ApJS 204, 7.
    arXiv: [1207.3845](https://arxiv.org/abs/1207.3845)
  - Almgren et al. (2020), *Castro: A Massively Parallel Compressible
    Astrophysics Simulation Code*, JOSS 5(54), 2513.
  - Zingale et al. (2019), Spectral deferred corrections in Castro,
    ApJ 886, 105. arXiv: [1908.03661](https://arxiv.org/abs/1908.03661)
- **Source:** <https://github.com/AMReX-Astro/Castro>
- **License:** BSD-3-Clause
- **Capabilities:** compressible Eulerian hydro (unsplit PPM, CTU);
  radiation hydrodynamics (gray and multigroup flux-limited diffusion);
  ideal MHD with constrained transport; nuclear reaction networks
  (via AMReX-Astro Microphysics); self-gravity (monopole, multigrid
  Poisson with isolated BCs); Helmholtz and stellar EOSs; rotating
  frames; point-mass gravity; block AMR with subcycling; GPU via
  AMReX (CUDA/HIP/SYCL); SDC time integration.
- **Dependencies:** AMReX, AMReX-Astro Microphysics, MPI, HDF5.

### 2.2 Nyx

Cosmological compressible hydro on AMReX.

- **Papers:**
  - Almgren et al. (2013), *Nyx: A Massively Parallel AMR Code for
    Computational Cosmology*, ApJ 765, 39. arXiv:
    [1301.4498](https://arxiv.org/abs/1301.4498)
  - Lukić et al. (2015), *The Lyman-α forest in optically thin
    hydrodynamical simulations*, MNRAS 446, 3697. arXiv:
    [1406.6361](https://arxiv.org/abs/1406.6361)
  - Sexton et al. (2021), JOSS software paper.
- **Source:** <https://github.com/AMReX-Astro/Nyx>
- **License:** BSD-3-Clause
- **Capabilities:** cosmological PPM hydro in comoving coordinates;
  dark-matter N-body (particle–mesh / CIC); self-gravity via geometric
  multigrid; primordial (H/He) heating/cooling with UV background;
  AMR with subcycling; GPU (CUDA/HIP/SYCL). Demonstrated on 13,824
  GPUs on Summit.

### 2.3 MAESTROeX

Low-Mach-number stratified stellar hydrodynamics.

- **Papers:**
  - Nonaka et al. (2010), ApJS 188, 358. arXiv:
    [1005.0112](https://arxiv.org/abs/1005.0112)
  - Fan et al. (2019), *MAESTROeX: A Massively Parallel Low Mach
    Number Astrophysical Solver*, ApJ 887, 212. arXiv:
    [1908.03634](https://arxiv.org/abs/1908.03634)
- **Source:** <https://github.com/AMReX-Astro/MAESTROeX>
- **License:** BSD-3-Clause
- **Capabilities:** pseudo-incompressible low-Mach stellar hydro with
  projection-based constraint; plane-parallel, full-star spherical,
  mapped geometries; nuclear reactions + thermal diffusion; Helmholtz
  / stellar EOSs; AMReX AMR + GPU.

### 2.4 Quokka

GPU-native two-moment radiation hydrodynamics on AMReX.

- **Paper:** Wibking & Krumholz (2022), *Quokka: A code for two-moment
  AMR radiation hydrodynamics on GPUs*, MNRAS 512, 1430. arXiv:
  [2110.01792](https://arxiv.org/abs/2110.01792)
- **Source:** <https://github.com/quokka-astro/quokka>
- **License:** MIT
- **Capabilities:** PPM hydro with MOL/SDC time integration; two-moment
  (M1 / variable Eddington) radiation transport with reduced speed of
  light; AMR with subcycling; self-gravity; tracer particles; chemistry
  hooks (Grackle); MHD in development; CUDA/HIP GPU-first.

### 2.5 FLASH / Flash-X

Mature multi-physics AMR instrument; Flash-X is the open-source
successor to FLASH4.

- **Papers:**
  - Fryxell et al. (2000), *FLASH: An Adaptive Mesh Hydrodynamics Code
    for Modeling Astrophysical Thermonuclear Flashes*, ApJS 131, 273.
  - Dubey et al. (2014), *Evolution of FLASH*, IJHPCA 28, 225.
  - Dubey et al. (2022), *Flash-X: A multiphysics simulation software
    instrument*, SoftwareX.
    DOI: [10.1016/j.softx.2022.101168](https://doi.org/10.1016/j.softx.2022.101168)
- **Source:** <https://github.com/Flash-X/Flash-X> (access by request to
  flash-x@lists.cels.anl.gov; code itself is open once granted)
- **License:** Apache-2.0 (legacy FLASH4 was under a custom academic
  license)
- **Capabilities:** unsplit/split PPM hydro; USM and Spark MHD;
  multigroup flux-limited diffusion radiation; nuclear networks + EOS
  (Helmholtz, multi-γ); self-gravity via multigrid/multipole/tree;
  HEDP (laser drive, multi-temperature plasma); incompressible and
  low-Mach reactive flow; Lagrangian particles, tracers, PIC, immersed
  boundary; AMR via Paramesh (legacy) or AMReX; GPU portability via
  the Flash-X Orchestration Runtime / Milhoja; integrates thornado for
  spectral M1 neutrino transport for CCSN.

### 2.6 GAMER / GAMER-2

GPU-accelerated patch-based AMR (NTU / Princeton).

- **Papers:**
  - Schive, Tsai, Chiueh (2010), *GAMER*, ApJS 186, 457. arXiv:
    [0907.3390](https://arxiv.org/abs/0907.3390)
  - Schive et al. (2018), *GAMER-2*, MNRAS 481, 4815. arXiv:
    [1712.07070](https://arxiv.org/abs/1712.07070)
- **Source:** <https://github.com/gamer-project/gamer>
- **License:** BSD-3-Clause
- **Capabilities:** octree-like patch AMR with adaptive timestepping;
  Hilbert SFC load balancing; hydro, MHD, special-relativistic hydro
  (GAMER-SR); self-gravity (multigrid + FFT root-grid Poisson); N-body
  + tracer particles; star formation; Grackle chemistry + cooling;
  cosmic rays with anisotropic diffusion; **fuzzy/wave dark matter
  (ELBDM)** — a distinguishing feature; cosmology; MPI + OpenMP + CUDA
  with bitwise reproducibility.

### 2.7 Athena++

Oct-tree block AMR (Princeton).

- **Papers:**
  - Stone et al. (2008), *Athena*, ApJS 178, 137. arXiv:
    [0804.0402](https://arxiv.org/abs/0804.0402) (predecessor)
  - Stone et al. (2020), *The Athena++ AMR Framework*, ApJS 249, 4.
    arXiv: [2005.06651](https://arxiv.org/abs/2005.06651)
  - Tomida & Stone (2023), Athena++ self-gravity multigrid. arXiv:
    [2302.13903](https://arxiv.org/abs/2302.13903)
  - Daszuta et al. (2024), GR-Athena++. arXiv:
    [2406.05126](https://arxiv.org/abs/2406.05126)
- **Source:** <https://github.com/PrincetonUniversity/athena>
- **License:** BSD-3-Clause
- **Capabilities:** oct-tree block AMR with task-list execution;
  Newtonian, SR, and GR hydro and MHD (constrained transport);
  curvilinear coords (Cartesian/cylindrical/spherical/GR); radiation
  transport (implicit short characteristics); self-gravity (multigrid +
  FFT); general EOS; super-time-stepping for diffusion; MPI + OpenMP
  (CPU — GPU path is AthenaK).

### 2.8 AthenaK

Kokkos port of Athena++ with full AMR (IAS / Princeton / MSU).

- **Papers:**
  - Stone et al. (2024), *AthenaK*, arXiv:
    [2409.16053](https://arxiv.org/abs/2409.16053)
  - Zhu et al. (2024), *Performance-Portable Numerical Relativity with
    AthenaK* (Z4c). arXiv: [2409.10383](https://arxiv.org/abs/2409.10383)
- **Source:** <https://github.com/IAS-Astrophysics/athenak>
- **License:** BSD-3-Clause
- **Capabilities:** multi-architecture GPU/CPU (Kokkos); block AMR;
  Newtonian / SR / GR hydro and MHD; GR in stationary and dynamical
  spacetimes; GR radiation MHD; numerical relativity via Z4c;
  Lagrangian tracers, charged test particles; ~1 × 10⁹ cell
  updates/s on a single GH200 and ≈ 80% efficiency on 65,536 Frontier
  GPUs.

### 2.9 Enzo & Enzo-E

- **Enzo:** The Enzo Collaboration (2014), *Enzo: An Adaptive Mesh
  Refinement Code for Astrophysics*, ApJS 211, 19. arXiv:
  [1307.2265](https://arxiv.org/abs/1307.2265). Source:
  <https://github.com/enzo-project/enzo-dev>. License: NCSA / modified
  BSD.
- **Enzo-E / Cello:** Bordner & Norman (2018), arXiv:
  [1810.01319](https://arxiv.org/abs/1810.01319). Source:
  <https://github.com/enzo-project/enzo-e>. License: multi-file
  permissive (NCSA + BSD-3-Clause + BSD-style).
- **Capabilities (Enzo):** Berger–Colella patch-based AMR; PPM, ZEUS,
  MUSCL hydro; Dedner and constrained-transport MHD; PM/PPM gravity
  with FFT root + multigrid subgrid; primordial + metal cooling (native
  and Grackle); ray-tracing and implicit FLD radiative transfer;
  Pop III / Pop II star-formation & feedback recipes; MPI+OpenMP CPU.
- **Capabilities (Enzo-E):** forest-of-octrees AMR built on Charm++
  over-decomposed runtime; PPM hydro (from Enzo), VL+CT (from
  Athena++), PPML MHD; scalable multigrid self-gravity; particles;
  Grackle cooling; designed for exascale; currently CPU-only.

### 2.10 K-Athena (archived)

First Kokkos port of Athena++. Grete, Glines, O'Shea (2021), IEEE TPDS
32, 85. arXiv: [1905.04341](https://arxiv.org/abs/1905.04341).
Source: <https://gitlab.com/pgrete/kathena>. License: BSD-3-Clause.
**Status:** superseded by AthenaK and by AthenaPK (Parthenon).

### 2.11 RAMSES

Octree AMR (Teyssier, Saclay / UZH).

- **Papers:**
  - Teyssier (2002), A&A 385, 337. arXiv:
    [astro-ph/0111367](https://arxiv.org/abs/astro-ph/0111367)
  - Fromang, Hennebelle, Teyssier (2006), A&A 457, 371 — CT MHD.
  - Rosdahl et al. (2013), RAMSES-RT, MNRAS 436, 2188. arXiv:
    [1304.7126](https://arxiv.org/abs/1304.7126)
- **Source:** <https://bitbucket.org/rteyssie/ramses>
- **License:** CeCILL-2.1 (GPL-compatible French free-software license)
- **Capabilities:** fully-threaded octree AMR with cell-level
  refinement; second-order unsplit Godunov/MUSCL; ideal MHD with
  constrained transport; multigrid + PM gravity on AMR; cosmology
  (FRW, zoom-in); RAMSES-RT multigroup M1 RT with non-equilibrium
  H/He chemistry; RAMSES-RTZ for non-equilibrium metal chemistry;
  atomic + metal cooling; Schmidt-law star formation; SN thermal and
  kinetic feedback; sink particles (AGN); MPI with Hilbert-curve
  decomposition; experimental GPU port (cuRAMSES).

### 2.12 PLUTO

Multi-physics Godunov code (Torino).

- **Papers:**
  - Mignone et al. (2007), *PLUTO*, ApJS 170, 228. arXiv:
    [astro-ph/0701854](https://arxiv.org/abs/astro-ph/0701854)
  - Mignone et al. (2012), *The PLUTO code for adaptive mesh
    computations*, ApJS 198, 7. arXiv:
    [1110.0740](https://arxiv.org/abs/1110.0740)
- **Source:** <http://plutocode.ph.unito.it/download.html> (registration-
  gated but free; several community GitHub mirrors exist)
- **License:** GPL (v2 / GPL-compatible)
- **Capabilities:** finite-volume Godunov for hyperbolic/parabolic
  PDEs; HD, ideal & resistive MHD, special-relativistic HD and MHD;
  radiative cooling; thermal conduction; viscosity; Hall and ambipolar
  MHD; multi-species / chemistry; dust (Lagrangian particles); hybrid
  PIC-MHD cosmic rays; Cartesian / cylindrical / spherical / polar;
  optional AMR via Chombo; gPLUTO GPU version.

### 2.13 MPI-AMRVAC

Octree AMR MHD framework (KU Leuven).

- **Papers:**
  - Keppens et al. (2012), JCP 231, 718.
  - Xia et al. (2018), *MPI-AMRVAC 2.0*, ApJS 234, 30.
  - Keppens et al. (2023), *MPI-AMRVAC 3.0*, A&A 673, A66. arXiv:
    [2303.03026](https://arxiv.org/abs/2303.03026)
- **Source:** <https://github.com/amrvac/amrvac>
- **License:** GPL-3.0
- **Capabilities:** block-based octree AMR for hyperbolic/parabolic
  PDEs; HD, ideal/resistive/Hall MHD, SR HD/MHD, two-fluid (ion–neutral)
  plasma, multi-fluid dust; radiative cooling; anisotropic thermal
  conduction; viscosity; gravity including multigrid self-gravity;
  super-time-stepping and IMEX integrators; synthetic EUV / X-ray
  observables on the fly.

### 2.14 ZEUS / ZEUS-MP (legacy)

Staggered-mesh MHD + FLD radiation from Stone & Norman (1992) and Hayes
et al. (2006). Source (mirror):
<https://github.com/bwoshea/ZEUS-MP_2>. License: NCSA-style. Active
development has ceased; cited primarily for historical reference and
as the archetype of Method-of-Characteristics constrained transport.

### 2.15 Dedalus

Spectral PDE framework (Burns, Vasil, Oishi, Lecoanet, Brown).

- **Paper:** Burns et al. (2020), *Dedalus: A Flexible Framework for
  Numerical Simulations with Spectral Methods*, PRR 2, 023068. arXiv:
  [1905.10388](https://arxiv.org/abs/1905.10388)
- **Source:** <https://github.com/DedalusProject/dedalus>
- **License:** GPL-3.0
- **Capabilities:** spectral discretization (Chebyshev, Fourier,
  Jacobi, disk/annulus/ball/shell); symbolic PDE entry; tau method for
  boundary conditions; IMEX time stepping; MPI parallel transposes.
  Excellent for smooth, spectrally-converging problems (anelastic/
  Boussinesq convection, rotating spherical convection, internal
  gravity waves, shear instabilities). Not a shock-capturing code.

---

## 3. Particle-Based and Meshless Codes

### 3.1 Arepo

Moving-mesh code (Springel et al., MPA / Harvard).

- **Papers:**
  - Springel (2010), *E pur si muove...*, MNRAS 401, 791. arXiv:
    [0901.4107](https://arxiv.org/abs/0901.4107)
  - Pakmor, Bauer, Springel (2011), MHD on moving mesh, MNRAS 418,
    1392. arXiv: [1108.1792](https://arxiv.org/abs/1108.1792)
  - Weinberger, Springel & Pakmor (2020), *The AREPO Public Code
    Release*, ApJS 248, 32. arXiv:
    [1909.04667](https://arxiv.org/abs/1909.04667)
- **Source:** <https://gitlab.mpcdf.mpg.de/vrs/arepo>
  (public release is a subset of the internal tree; IllustrisTNG /
  Auriga / SMUGGLE feedback modules remain private)
- **License:** GPL-3.0-or-later
- **Capabilities:** finite-volume Godunov on a moving Voronoi mesh
  (quasi-Lagrangian, Galilean-invariant); ideal MHD with Powell 8-wave
  cleaning; TreePM gravity (Newtonian and comoving); cosmological
  integrator; primordial cooling + Springel–Hernquist subgrid ISM/SF
  (public) — TNG adds BH seeding, AGN thermal+kinetic feedback,
  chemical enrichment, metal-line cooling; MPI distributed domain
  decomposition with dynamic balancing; no native GPU support.

### 3.2 GADGET-4

- **Paper:** Springel, Pakmor, Zier, Reinecke (2021), *Simulating
  cosmic structure formation with the GADGET-4 code*, MNRAS 506, 2871.
  arXiv: [2010.03567](https://arxiv.org/abs/2010.03567)
- **Source:** <https://gitlab.mpcdf.mpg.de/vrs/gadget4>
- **License:** GPL-3.0-or-later
- **Capabilities:** modern pressure–entropy SPH; tree, TreePM, and
  Fast Multipole Method (FMM) gravity with hierarchical time-stepping;
  cosmology with 2LPT IC generator; lightweight built-in cooling + SF;
  on-the-fly FOF and SUBFIND-HBT halo finders; integrated merger-tree
  builder; high dynamic range power-spectrum estimator; light-cones;
  MPI + MPI-3 shared-memory windows; no GPU.

### 3.3 GIZMO

- **Papers:**
  - Hopkins (2015), *GIZMO: A New Class of Accurate, Mesh-Free
    Hydrodynamic Simulation Methods*, MNRAS 450, 53. arXiv:
    [1409.7395](https://arxiv.org/abs/1409.7395)
  - Hopkins & Raives (2016), MHD for meshless methods, MNRAS 455, 51.
    arXiv: [1505.02783](https://arxiv.org/abs/1505.02783)
  - Hopkins et al. (2018), FIRE-2 physics, MNRAS 480, 800. arXiv:
    [1702.06148](https://arxiv.org/abs/1702.06148)
- **Source:** <https://bitbucket.org/phopkins/gizmo-public> (FIRE,
  STARFORGE, SMUGGLE extensions are private)
- **License:** GPL (public distribution)
- **Capabilities:** Meshless Finite-Mass (MFM), Meshless Finite-Volume
  (MFV), modern SPH (pressure–energy, DISPH), fixed-grid option; ideal
  and non-ideal MHD (Ohmic, Hall, ambipolar); Tree–PM gravity with
  adaptive softening; cosmology; radiative cooling (Grackle / in-house
  tables); anisotropic conduction and viscosity; cosmic-ray transport
  (diffusion, streaming, Alfvén coupling); dust dynamics; radiation
  hydrodynamics (LEBRON / M1 / FLD); nuclear reaction networks; sink
  particles (STARFORGE); MPI + OpenMP (limited GPU in public version).

### 3.4 ChaNGa

- **Papers:**
  - Menon et al. (2015), *Adaptive techniques for clustered N-body
    cosmological simulations*, CompAC 2, 1. arXiv:
    [1409.1929](https://arxiv.org/abs/1409.1929)
  - Jetley et al. (2010), *Scaling Hierarchical N-body Simulations on
    GPU Clusters*, SC '10.
- **Source:** <https://github.com/N-BodyShop/changa>
- **License:** GPL-2.0
- **Capabilities:** Barnes–Hut tree with hexadecapole multipoles,
  Ewald summation for periodic BCs; GASOLINE-style SPH with modern
  variants (geometric density, turbulent diffusion); cosmology; cooling
  (primordial + metal-line), SF, SN feedback, UV background, black
  holes; CUDA-accelerated gravity tree-walk; Charm++ message-driven
  runtime with dynamic load balancers; demonstrated at 512 K cores on
  Blue Waters.

### 3.5 SWIFT

- **Papers:**
  - Schaller et al. (2024), *SWIFT...*, MNRAS 530, 2378. arXiv:
    [2305.13380](https://arxiv.org/abs/2305.13380)
  - Borrow et al. (2022), *SPHENIX SPH in SWIFT*. arXiv:
    [2012.03957](https://arxiv.org/abs/2012.03957)
  - Elbers et al. (2021), δf-neutrinos. arXiv:
    [2010.07321](https://arxiv.org/abs/2010.07321)
- **Source:** <https://gitlab.cosma.dur.ac.uk/swift/swiftsim> (mirror
  <https://github.com/SWIFTSIM/SWIFT>)
- **License:** LGPL-3.0-or-later
- **Capabilities:** multiple swappable SPH schemes (Minimal, Gadget-2,
  SPHENIX, Gasoline-2, AnarchyPU, Phantom) and Meshless Finite-Mass;
  FMM gravity with periodic Ewald via mesh (FMM–PM); cosmology with
  on-the-fly power spectra and light-cones; massive neutrinos via δf
  particle method; pluggable subgrid stacks (EAGLE, EAGLE-XL, COLIBRE,
  GEAR, AGORA) covering cooling, SF, stellar/SN/AGN feedback, chemical
  enrichment; GEAR-RT and SPH-M1-RT radiative transfer; BH seeding via
  FOF; task-based parallelism with dependency graph; hybrid
  MPI + pthreads + SIMD; experimental GPU/OpenACC/SYCL ports.

---

## 4. Relativistic / GRMHD / Numerical-Relativity Codes

### 4.1 Phoebus

Radiation GRMHD on Parthenon (LANL).

- **Paper:** Barrows, Miller, Ryan et al. (2024), *Phoebus: Performance
  Portable GRRMHD for Relativistic Astrophysics*. arXiv:
  [2410.09146](https://arxiv.org/abs/2410.09146)
- **Source:** <https://github.com/lanl/phoebus>
- **License:** BSD-3-Clause
- **Capabilities:** GRMHD with constrained transport on fixed
  (stationary) background spacetime; neutrino transport via Monte-Carlo
  and moment methods; opacities via Singularity-Opac; tabulated nuclear
  / finite-T EOS via Singularity-EOS; block AMR via Parthenon;
  GPU-resident via Kokkos (CUDA/HIP/SYCL/OpenMP); primitive recovery
  tailored to GRMHD + radiation.

### 4.2 Einstein Toolkit

Open community NR/RMHD infrastructure built on Cactus.

- **Papers:**
  - Löffler et al. (2012), *The Einstein Toolkit*, CQG 29, 115001.
    arXiv: [1111.3344](https://arxiv.org/abs/1111.3344)
  - Mösta et al. (2014), *GRHydro*, CQG. arXiv:
    [1304.5544](https://arxiv.org/abs/1304.5544)
- **Source:** <https://einsteintoolkit.org/> (GitHub org
  <https://github.com/EinsteinToolkit>, Bitbucket mirror)
- **License:** Cactus flesh + core thorns LGPL; individual thorns
  carry their own licenses (GPL/LGPL/BSD) — no single SPDX.
- **Capabilities:** dynamical spacetime evolution (BSSN via McLachlan,
  Z4c); GRHD (GRHydro, IllinoisGRMHD) and GRMHD with vector-potential
  constrained transport; neutrino leakage (ZelmaniLeak) and M1 via
  extensions (Spritz, WhiskyTHC); AMR via Carpet (CPU) or CarpetX
  (AMReX-backed, GPU); tabulated finite-T nuclear EOS (EOS_Omni); a
  variety of primitive recovery schemes (2D Noble, Palenzuela-Newman,
  RePrimAnd).

### 4.3 SpECTRE

Discontinuous Galerkin NR/GRMHD code (SXS Collaboration).

- **Papers:**
  - Kidder et al. (2017), *SpECTRE*, JCP 335, 84. arXiv:
    [1609.00098](https://arxiv.org/abs/1609.00098)
  - Deppe et al. (2022), DG–FD hybrid for GRMHD. arXiv:
    [2109.11645](https://arxiv.org/abs/2109.11645)
- **Source:** <https://github.com/sxs-collaboration/spectre>
- **License:** MIT
- **Capabilities:** nodal DG with DG–FD hybrid shock-capturing;
  dynamical spacetime (Generalized Harmonic; CCZ4 in development);
  GRMHD targeting BNS/BHNS; M1 neutrino transport under active
  development; task-based h/p-adaptive AMR via Charm++; analytic and
  tabulated nuclear EOS; multiple primitive-recovery schemes. GPU
  support is experimental.

### 4.4 BHAC — Black Hole Accretion Code

GRMHD on MPI-AMRVAC (Frankfurt / Radboud).

- **Papers:**
  - Porth et al. (2017), *The Black Hole Accretion Code*, CompAC 4, 1.
    arXiv: [1611.09720](https://arxiv.org/abs/1611.09720)
  - Olivares et al. (2019), constrained transport + AMR in BHAC. arXiv:
    [1906.10795](https://arxiv.org/abs/1906.10795)
  - Ripperda et al. (2019), GR resistive MHD. arXiv:
    [1907.07197](https://arxiv.org/abs/1907.07197)
- **Source:** <https://bhac.science/>, GitLab:
  <https://gitlab.itp.uni-frankfurt.de/BHAC-release/bhac>
- **License:** GPL-3.0-or-later
- **Capabilities:** ideal and resistive GRMHD on arbitrary stationary
  spacetimes; 1/2/3D; octree block AMR via MPI-AMRVAC; upwind CT /
  flux-CT; CPU MPI; ideal-gas / polytropic / piecewise polytropic EOS;
  1D/2D Noble + Palenzuela primitive recovery.

### 4.5 KORAL

Radiative GRMHD (Sądowski; Chael's `koral_lite`).

- **Papers:**
  - Sądowski et al. (2013), semi-implicit M1 radiation in GR. arXiv:
    [1212.5050](https://arxiv.org/abs/1212.5050)
  - Sądowski et al. (2014), super-critical accretion. arXiv:
    [1311.5900](https://arxiv.org/abs/1311.5900)
  - Chael, Narayan, Johnson (2019), two-temperature GRRMHD of M87.
    arXiv: [1810.01983](https://arxiv.org/abs/1810.01983)
- **Source:** <https://github.com/achael/koral_lite> (original private)
- **License:** GPL-3.0
- **Capabilities:** M1-closure radiative GRMHD; two-temperature
  (ion/electron) thermodynamics; non-thermal electron evolution;
  Compton, synchrotron, bremsstrahlung opacities; fixed background
  spacetime; static multi-patch grids (no block AMR); CPU (MPI +
  OpenMP).

### 4.6 HARM / HARM3D / iharm3d

GRMHD lineage (Gammie, Noble, Tchekhovskoy, UIUC AFD).

- **Papers:**
  - Gammie, McKinney, Tóth (2003), *HARM*, ApJ 589, 444. arXiv:
    [astro-ph/0301509](https://arxiv.org/abs/astro-ph/0301509)
  - Noble et al. (2006), primitive-variable solvers, ApJ 641, 626.
    arXiv: [astro-ph/0512420](https://arxiv.org/abs/astro-ph/0512420)
  - Prather et al. (2021), *iharm3D: Vectorized GRMHD*, JOSS. arXiv:
    [2110.10191](https://arxiv.org/abs/2110.10191)
  - Murguia-Berthier et al. (2021), HARM3D+NUC (tabulated EOS +
    neutrino leakage). arXiv:
    [2106.05356](https://arxiv.org/abs/2106.05356)
- **Source:** iharm3d <https://github.com/AFD-Illinois/iharm3d>,
  HARMPI <https://github.com/atchekho/harmpi>, HARM_COOL variants.
- **License:** iharm3d GPL-3.0; other forks vary — HARM3D+NUC not
  publicly released.
- **Capabilities:** ideal GRMHD with Tóth flux-CT; fixed stationary
  spacetime; Kerr / Kerr-Schild / MKS / FMKS coordinates; uniform
  logical grid (no AMR); MPI + OpenMP; iharm3d adds SIMD; tabulated
  nuclear EOS + neutrino leakage available only in +NUC fork.

### 4.7 KHARMA

Kokkos/Parthenon GRMHD (UIUC AFD).

- **Paper:** Prather (2024), *KHARMA: Flexible, Portable Performance
  for GRMHD*, SMC 2024. arXiv:
  [2408.01361](https://arxiv.org/abs/2408.01361)
- **Source:** <https://github.com/AFD-Illinois/kharma>
- **License:** BSD-3-Clause
- **Capabilities:** ideal GRMHD with flux-CT; modular packages;
  fixed stationary spacetime with many coordinate systems; block
  SMR/AMR via Parthenon; full GPU portability via Kokkos; runtime-
  swappable primitive recovery (2D Noble, 1D_w, ONED_W); electron
  temperature and viscous-hydro extensions; used in EHT production.

### 4.8 GRChombo

Numerical relativity with AMR via Chombo.

- **Papers:**
  - Clough et al. (2015), *GRChombo*, CQG 32, 245011. arXiv:
    [1503.03436](https://arxiv.org/abs/1503.03436)
  - Andrade et al. (2022), JOSS 7, 3703. arXiv:
    [2201.03458](https://arxiv.org/abs/2201.03458)
- **Source:** <https://github.com/GRTLCollaboration/GRChombo>
- **License:** BSD-3-Clause
- **Capabilities:** dynamical spacetime (CCZ4 / BSSN, moving-puncture)
  with arbitrary matter; specialized for fundamental physics (scalar
  fields, dark matter, modified gravity via GRFolres); Berger–Rigoutsos
  block AMR via Chombo; MPI + OpenMP + SIMD; no first-class GRMHD.

### 4.9 Spritz, WhiskyTHC (Einstein Toolkit thorns)

- **Spritz** — Cipolletta et al. (2020/2021) — full GRMHD on
  dynamical spacetime via vector-potential CT with neutrino leakage;
  GPLv2+. arXiv: [1912.04794](https://arxiv.org/abs/1912.04794),
  [2012.10174](https://arxiv.org/abs/2012.10174).
- **WhiskyTHC** — Radice, Rezzolla et al. (2012/2014) — high-order
  FD/WENO GR hydrodynamics with M1 gray/energy-integrated neutrino
  transport; arXiv: [1206.6502](https://arxiv.org/abs/1206.6502),
  [1312.5004](https://arxiv.org/abs/1312.5004). Distribution is
  by collaboration agreement rather than open-source release.

---

## 5. Stellar Structure / Evolution and CCSN Microphysics Codes

### 5.1 MESA

- **Papers:** instrument series — Paxton et al. (2011 ApJS 192, 3;
  2013 ApJS 208, 4; 2015 ApJS 220, 15; 2018 ApJS 234, 34; 2019 ApJS
  243, 10); Jermyn et al. (2023 ApJS 265, 15).
- **Source:** <https://github.com/MESAHub/mesa> —
  <https://mesastar.org/>
- **License:** LGPL-3.0
- **Capabilities:** 1-D Lagrangian stellar evolution from PMS to
  WD/NS/BH (0.01–1000 M⊙); coupled hydrodynamics + nuclear burning
  with adaptive large networks; shellular rotation; Tayler–Spruit
  magnetic torques; element diffusion and gravitational settling;
  MLT, TDC, semiconvection, overshooting, thermohaline mixing;
  binaries (mass transfer, simplified CE, tidal sync); asteroseismic
  output for GYRE; explicit hydrodynamics for explosions/pulsations;
  implicit solver with adaptive mesh and timestep; ships its own
  toolchain (MESA SDK).

### 5.2 KEPLER

Foundational 1-D Lagrangian stellar hydrodynamics + very large nuclear
networks (Weaver/Zimmerman/Woosley). Papers include Weaver, Zimmerman,
Woosley (1978), Rauscher et al. (2002), Woosley, Heger & Weaver (2002).
Documentation: <https://2sn.org/kepler/>. **Not publicly released**;
access by arrangement with the Woosley/Heger groups. Gold standard
for massive-star nucleosynthesis (s/r-process yields, presupernova
models).

### 5.3 CCSN / neutrino-transport codes

- **CHIMERA** — Bruenn et al. (2020), ApJS 248, 11. Ray-by-ray-plus
  multi-group FLD neutrino transport + PPM hydro + effective-GR
  monopole gravity; 2D/3D (Yin-Yang) spherical. **Restricted**
  (collaboration-only).
- **FORNAX** — Skinner et al. (2019), ApJS 241, 7. Multi-group
  M1 neutrino RHD to O(v/c); 3 species; static dendritic spherical
  mesh; effective-GR monopole gravity; realistic nuclear EOS.
  **Not publicly released**.
- **thornado** — Endeve, Pochik, Laiu, Barker et al. Discontinuous
  Galerkin spectral two-moment M1 neutrino transport (6 species);
  SR observer corrections to O(v/c); IMEX time-stepping; GPU via
  OpenMP-offload / OpenACC; integrates into Flash-X for CCSN, and
  runs standalone. Key references: Laiu et al. (2020), arXiv:
  [2009.05617](https://arxiv.org/abs/2009.05617); Couch, Dubey,
  Endeve, Harris et al. (2026), *thornado+Flash-X: A Hybrid DG-IMEX
  and Finite-Volume Framework for Neutrino-Radiation Hydrodynamics
  in Core-Collapse Supernovae*, arXiv:
  [2601.00976](https://arxiv.org/abs/2601.00976). Source
  <https://github.com/endeve/thornado>, license: BSD-3-Clause.
- **WeakLib** — companion tabulated nuclear EOS + neutrino opacities
  library.

---

## 6. Union of Capabilities

The following list enumerates every capability that appears in at
least one of the codes above. A simulation engine aiming to be able
to perform the tasks of any of these codes would need, in aggregate,
the following:

### 6.1 Discretizations and meshes

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

### 6.2 Hyperbolic / fluid physics

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

### 6.3 Radiation, transport, and chemistry

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

### 6.4 Gravity and cosmology

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

### 6.5 Equations of state and microphysics

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

### 6.6 Subgrid / astrophysical "physics modules"

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

### 6.7 Stellar evolution

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

### 6.8 Numerics / solver technology

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

### 6.9 Parallelism and performance portability

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

### 6.10 I/O, diagnostics, and ecosystem

- Parallel HDF5 and PnetCDF; AMReX plotfiles; ADIOS2; VTK/yt-
  compatible output.
- On-the-fly synthetic observables (EUV, X-ray, spectra).
- Integrated analysis hooks (yt, VisIt, ParaView, Ascent).
- On-the-fly halo-finding, merger trees, light-cones, power spectra.

### 6.11 Licensing and openness landscape

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

---

## 7. Implications for Cosmic Foundry

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
     and SYCL execution — conceptually a Kokkos analogue — so that
     kernels written once compile and run across vendor GPUs and
     CPUs.
   - A task-based asynchronous driver with explicit dependency
     graphs, over-decomposition, and dynamic load balancing (the
     role played elsewhere by Athena++'s task list, Parthenon's
     driver, SWIFT's task graph, and Charm++).
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

## 7. Implications for Cosmic Foundry

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
