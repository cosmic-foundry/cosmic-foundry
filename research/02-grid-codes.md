# 2. Structured-Grid / AMR Finite-Volume Codes

> Part of the [Cosmic Foundry research notes](index.md).

## 2.1 Castro

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

## 2.2 Nyx

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

## 2.3 MAESTROeX

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

## 2.4 Quokka

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

## 2.5 FLASH / Flash-X

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

## 2.6 GAMER / GAMER-2

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

## 2.7 Athena++

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

## 2.8 AthenaK

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

## 2.9 Enzo & Enzo-E

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

## 2.10 K-Athena (archived)

First Kokkos port of Athena++. Grete, Glines, O'Shea (2021), IEEE TPDS
32, 85. arXiv: [1905.04341](https://arxiv.org/abs/1905.04341).
Source: <https://gitlab.com/pgrete/kathena>. License: BSD-3-Clause.
**Status:** superseded by AthenaK and by AthenaPK (Parthenon).

## 2.11 RAMSES

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

## 2.12 PLUTO

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

## 2.13 MPI-AMRVAC

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

## 2.14 ZEUS / ZEUS-MP (legacy)

Staggered-mesh MHD + FLD radiation from Stone & Norman (1992) and Hayes
et al. (2006). Source (mirror):
<https://github.com/bwoshea/ZEUS-MP_2>. License: NCSA-style. Active
development has ceased; cited primarily for historical reference and
as the archetype of Method-of-Characteristics constrained transport.

## 2.15 Dedalus

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
