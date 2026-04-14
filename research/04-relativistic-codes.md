# 4. Relativistic / GRMHD / Numerical-Relativity Codes

> Part of the [Cosmic Foundry research notes](index.md).

## 4.1 Phoebus

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

## 4.2 Einstein Toolkit

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

## 4.3 SpECTRE

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

## 4.4 BHAC — Black Hole Accretion Code

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

## 4.5 KORAL

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

## 4.6 HARM / HARM3D / iharm3d

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

## 4.7 KHARMA

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

## 4.8 GRChombo

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

## 4.9 Spritz, WhiskyTHC (Einstein Toolkit thorns)

- **Spritz** — Cipolletta et al. (2020/2021) — full GRMHD on
  dynamical spacetime via vector-potential CT with neutrino leakage;
  GPLv2+. arXiv: [1912.04794](https://arxiv.org/abs/1912.04794),
  [2012.10174](https://arxiv.org/abs/2012.10174).
- **WhiskyTHC** — Radice, Rezzolla et al. (2012/2014) — high-order
  FD/WENO GR hydrodynamics with M1 gray/energy-integrated neutrino
  transport; arXiv: [1206.6502](https://arxiv.org/abs/1206.6502),
  [1312.5004](https://arxiv.org/abs/1312.5004). Distribution is
  by collaboration agreement rather than open-source release.
