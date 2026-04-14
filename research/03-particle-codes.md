# 3. Particle-Based and Meshless Codes

> Part of the [Cosmic Foundry research notes](index.md).

## 3.1 Arepo

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

## 3.2 GADGET-4

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

## 3.3 GIZMO

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

## 3.4 ChaNGa

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

## 3.5 SWIFT

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
