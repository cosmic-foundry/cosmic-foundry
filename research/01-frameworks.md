# 1. Frameworks & Infrastructure Libraries

> Part of the [Cosmic Foundry research notes](index.md).

## 1.1 AMReX

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

## 1.2 Kokkos

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

## 1.3 Parthenon

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

## 1.4 Charm++

Message-driven, over-decomposed asynchronous runtime used by ChaNGa,
Enzo-E, and SpECTRE. (Not a simulation library itself, but the
parallelism model of three relevant codes.) Source:
<https://github.com/charmplusplus/charm>, license: University of
Illinois / NCSA Open Source License.

## 1.5 Other supporting libraries

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
