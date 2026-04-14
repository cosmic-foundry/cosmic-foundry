# 5. Stellar Structure / Evolution and CCSN Microphysics Codes

> Part of the [Cosmic Foundry research notes](index.md).

## 5.1 MESA

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

## 5.2 KEPLER

Foundational 1-D Lagrangian stellar hydrodynamics + very large nuclear
networks (Weaver/Zimmerman/Woosley). Papers include Weaver, Zimmerman,
Woosley (1978), Rauscher et al. (2002), Woosley, Heger & Weaver (2002).
Documentation: <https://2sn.org/kepler/>. **Not publicly released**;
access by arrangement with the Woosley/Heger groups. Gold standard
for massive-star nucleosynthesis (s/r-process yields, presupernova
models).

## 5.3 CCSN / neutrino-transport codes

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
