# Computational Astrophysics Code Landscape — Research Notes

This document surveys major computational astrophysics codes. For each
code it records canonical papers, public source location, license, and
capabilities. The survey is organized by infrastructure family rather
than alphabetically, because most modern codes cluster around a small
set of framework design patterns.

---

# 1. Frameworks & Infrastructure Libraries

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

AMReX's GPU portability model is built around `ParallelFor`, a flat
iteration abstraction: physics authors write a per-cell lambda; AMReX
launches it as a GPU kernel or CPU loop depending on the build target.
AMReX does **not** expose thread-team or scratchpad abstractions to
physics authors. Performance on memory-hierarchy-sensitive operations
(multigrid, FFT, sparse solvers) comes from AMReX's own pre-tuned
framework kernels and from delegating to vendor libraries (cuFFT,
cuSolver, Hypre, PETSc). Physics codes built on AMReX (Castro, Quokka,
Flash-X) consequently write their physics as flat per-cell lambdas and
rely on AMReX or vendor libraries for the operations where explicit
shared-memory tiling matters.

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

### Hierarchical execution model

Kokkos's performance-portability guarantee rests on a **hierarchical
execution policy** that mirrors the execution and memory hierarchy of
every supported target. There are three distinct execution policies:

**`RangePolicy`** — flat 1-D iteration. One thread per index. Maps to a
CUDA kernel (one thread per element), an OpenMP `parallel for`, or a
serial loop. The default for elementwise operations with no neighbor
dependencies.

**`MDRangePolicy`** — multi-dimensional tiled iteration. Tile dimensions
are tunable per backend to match cache-line size, SIMD width, or GPU
warp size. Used for structured-grid loops where data layout tiling
improves reuse without requiring explicit shared memory.

**`TeamPolicy`** — a grid of *thread teams*, each team cooperative. One
team maps to a CUDA thread block, an OpenMP parallel region within an
outer `parallel for`, or an inner loop in serial. This is the policy
that enables portable shared-memory tiling.

Within a `TeamPolicy` kernel, the team has access to a scratchpad
declared at launch time:

```cpp
Kokkos::TeamPolicy<>( N_tiles, team_size )
    .set_scratch_size( 0, Kokkos::PerTeam( sizeof(double) * (tile + 2*halo) ) )
```

Inside the kernel body, scratch is accessed as a typed `View` in
`ScratchMemorySpace`. On CUDA this compiles to `__shared__` memory; on
OpenMP it allocates per-team on the thread stack; in serial it is a
plain stack array. The physics author writes the same source regardless.

The canonical tiled-stencil pattern using `TeamPolicy`:

```cpp
KOKKOS_LAMBDA(const team_t& team) {
  // Allocate team scratchpad (→ __shared__ on CUDA)
  Kokkos::View<double*, ScratchSpace> smem(team.team_scratch(0),
                                           tile + 2*halo);
  // Cooperative load: all threads in this team fill the halo region
  Kokkos::parallel_for(Kokkos::TeamThreadRange(team, tile + 2*halo),
    [&](int i){ smem(i) = q(tile_start - halo + i); });

  team.team_barrier();   // → __syncthreads() on CUDA; no-op on CPU

  // Per-thread computation reads from scratch
  Kokkos::parallel_for(Kokkos::TeamThreadRange(team, tile),
    [&](int i){ result(tile_start + i) = stencil(smem, i); });
}
```

`team.team_barrier()` maps to `__syncthreads()` on CUDA and to a
no-op or a lightweight fence elsewhere. Without it, threads that finish
the load early could begin reading scratch before other threads have
written their portion.

### Memory hierarchy

Kokkos exposes four levels, each with a corresponding `MemorySpace`:

| Level | Kokkos space | CUDA analog | Typical latency |
|---|---|---|---|
| Global (HBM) | `MemorySpace` (default) | device global memory | ~600 cycles |
| Team scratch L0 | `ScratchMemorySpace` level 0 | `__shared__` (fast SRAM) | ~20 cycles |
| Team scratch L1 | `ScratchMemorySpace` level 1 | L1/extended shared | ~80 cycles |
| Per-thread scratch | `PerThread` scratch | registers / local memory | ~1–4 cycles |

The practical implication: a stencil that loads its halo into team
scratch L0 before computing makes ~30× fewer HBM accesses than one
that reads global memory per thread. For bandwidth-bound operations
(reconstruction, divergence, Laplacian) on large domains this is the
dominant performance lever.

### How Kokkos differs from AMReX

AMReX exposes only the `RangePolicy` equivalent to physics authors.
Kokkos exposes all three levels and requires the physics author to
choose the right policy and manage scratch explicitly. The tradeoff:
Kokkos gives more control and portable access to the memory hierarchy
at the cost of a more complex programming model. Downstream codes
reflect this: Parthenon/AthenaK physics kernels are written with
awareness of team size, tile dimensions, and scratch layout;
AMReX-based physics kernels are generally flat per-cell lambdas.

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

Parthenon uses two distinct mechanisms that are easy to conflate but
solve different problems:

**Task-based driver (`TaskList` / `TaskRegion`).** A timestep is
represented as a dependency graph of coarse tasks (fill ghost zones,
compute fluxes, apply conservative update). The runtime walks the
graph and dispatches tasks as their dependencies are satisfied. The
primary purpose is *latency hiding*: MPI communication for one
meshblock can proceed while compute tasks execute on another. This is
*not* a kernel-granularity mechanism; each task may still dispatch
multiple kernel launches.

**Variable/meshblock packing (`MeshBlockPack`).** AMR produces many
small meshblocks. Launching one CUDA kernel per block per operation
incurs prohibitive kernel-launch overhead and leaves GPUs
underoccupied. Parthenon addresses this by packing multiple meshblocks
(and optionally multiple variables) into a single contiguous device
allocation, then issuing one `ParallelFor` over the entire pack.
Kernel-launch granularity is controlled by the packing layer. Per-cell
operations — EOS evaluations, Riemann solvers, reconstruction — are
written as ordinary device-callable functions inlined into the
surrounding `ParallelFor`; they are unaware of the pack size.

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

## 1.6 GPU kernel DSLs: Singe and related systems

This section covers domain-specific languages and compiler systems
that target GPU kernel generation for irregular or chemistry-heavy
scientific workloads. The motivating application is combustion
chemistry, but the techniques are structurally identical to nuclear
reaction networks in astrophysics (§6.5) and the kernel-abstraction
problem this engine must solve.

### Singe

Domain-specific language for combustion chemistry on GPUs (Stanford
— Bauer, Treichler, and Aiken). From the same research group as the
Legion task-based runtime (§1.6 below).

- **Paper:** Bauer, Treichler, Aiken (2014), *Singe: Leveraging Warp
  Specialization for High Performance on GPUs*, PPoPP 2014. arXiv /
  PDF: <https://cs.stanford.edu/~sjt/pubs/ppopp14.pdf>
- **License:** research prototype; no public production release.

#### The warp-specialization model

Standard SIMT execution requires that all threads within a warp
execute the same instruction at each clock cycle. Divergence within a
warp (threads taking different branches) serializes the warp and
reduces throughput proportionally. The traditional response is to
ensure all threads in a warp take the same branch — but for irregular
workloads such as stiff ODE systems, this is difficult: different
cells require different numbers of Newton iterations to converge, and
there is no a priori way to guarantee all warps finish at the same
time.

Singe's response is **warp specialization**: rather than making all
warps execute the same code path, different warps within a thread
block are assigned to different *roles* at kernel launch time. Since
SIMT divergence is only a penalty within a single warp (not across
warps), different warps can follow arbitrarily different code paths
with no serialization overhead. The divergence is shifted from
thread-level (expensive) to warp-level (free).

In the combustion context, warp roles correspond to different
computational paths through the chemistry solver. Cells requiring few
solver iterations are processed by one set of warps; cells requiring
many iterations by another. Within each warp, all threads execute the
same path, so SIMT efficiency is preserved. The block as a whole
covers the full range of stiffness without any warp waiting on others.

The Singe compiler takes a high-level description of the computation
and generates the warp-specialized CUDA kernel, including the
assignment of warps to roles and any required communication between
them. The physics author does not write CUDA directly.

#### Relationship to other execution models

| Model | Divergence unit | Physics-author visibility |
|---|---|---|
| Flat `RangePolicy` | Thread (expensive) | No control |
| Kokkos `TeamPolicy` | Thread within team (expensive); warps in team execute same code | Team-level control |
| Singe warp specialization | Warp (free) | Role declared in DSL; compiler assigns |

Warp specialization is strictly more expressive than `TeamPolicy`:
in `TeamPolicy`, all threads in a team execute the same instruction
via `TeamThreadRange`. In warp specialization, different warps in the
same block execute different code. The `TeamPolicy` model cannot
express this; it requires that threads differ only in their index,
not in their code path.

#### Relevance to nuclear reaction networks

Nuclear reaction networks in stellar astrophysics (aprox13, aprox19,
XNet, AMReX-Astro Microphysics) are structurally identical to
combustion reaction networks: both are stiff coupled ODEs, both have
per-cell computational cost that varies by orders of magnitude
depending on local thermodynamic conditions, and both are evaluated
at every cell on every timestep, making them a dominant hot path.
The load-imbalance and SIMT-divergence problems Singe addresses are
therefore directly relevant to the microphysics layer of this engine
(Epoch 6).

### PyJac

Analytical Jacobian generator for combustion chemistry mechanisms
(Niemeyer, Sung, Raju, 2017).

- **Paper:** Niemeyer et al. (2017), *pyJac: analytical Jacobian
  generator for chemical kinetics*, CPC 215, 188. arXiv:
  [1605.03262](https://arxiv.org/abs/1605.03262)
- **Source:** <https://github.com/SLACKHA/pyJac>
- **License:** MIT

Rather than evaluating the chemistry Jacobian by finite differences
(O(N_species) extra RHS evaluations per Newton step), PyJac generates
analytical Jacobian expressions from the mechanism description using a
symbolic code generator and emits optimized C or CUDA source. This
reduces Jacobian cost from O(N²) to O(N) for sparse mechanisms and
is the primary technique enabling fast implicit solvers for large
networks on GPUs.

### TChem

GPU-accelerated thermochemistry and kinetics toolkit (Sandia National
Laboratories).

- **Source:** <https://github.com/sandialabs/TChem>
- **License:** BSD 3-Clause
- **Backend:** Kokkos (CUDA, HIP, OpenMP, Serial)

TChem implements combustion reaction-network ODE integration using
Kokkos's hierarchical execution model. It exercises `TeamPolicy` with
team-level scratch for the Newton solve and demonstrates how the
Kokkos memory hierarchy (§1.2) applies in practice to stiff chemistry.
It is the closest available open-source reference for how a
Kokkos-based engine would implement the microphysics hot path.

### Legion / Regent

The same group that produced Singe also built the broader Legion
programming system (Bauer, Treichler, Slaughter, Aiken, Stanford /
LANL / NVIDIA):

- **Legion** — task-based runtime for distributed heterogeneous
  architectures. Separates program *correctness* (task dependencies,
  data partitioning) from *performance* (mapping tasks to processors,
  data placement). Source: <https://github.com/StanfordLegion/legion>,
  license: Apache-2.0.
- **Regent** — high-productivity language built on Legion. Provides
  implicit parallelism extraction from sequential-looking code.
  Source: <https://github.com/StanfordLegion/legion/tree/master/language>

Legion's algorithm / mapping separation — where the same logical
program can be remapped to different hardware by changing only the
mapper, not the computation — is the distributed-memory analog of
Singe's computation / warp-assignment separation. Both systems
separate *what* to compute from *how* to organize threads or tasks.

---

# 2. Structured-Grid / AMR Finite-Volume Codes

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

---

# 3. Particle-Based and Meshless Codes

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

---

# 4. Relativistic / GRMHD / Numerical-Relativity Codes

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

---

# 5. Stellar Structure / Evolution and CCSN Microphysics Codes

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

---

# 6. Union of Capabilities

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

## 6.11 Visualization and science communication

Visualization in the surveyed codes is almost entirely batch,
post-hoc, and desktop-shaped — plotfiles and HDF5 go out, analysis
frameworks (yt, VisIt, ParaView, Ascent) come in. The landscape
relevant to a modern engine is broader than that traditional HPC
analysis stack.

**Python analysis ecosystem.**

- `yt` (BSD-3) — de facto analysis standard for AMR and particle
  astrophysics output; unit-aware; Jupyter widgets via `widgyts`;
  supports AMReX, Enzo, Flash, GADGET, Arepo, RAMSES outputs.
- `napari` (BSD-3) — multi-dimensional image viewer, originally
  bio-imaging; useful for slice stacks and labeled volumes.
- `pyvista` (MIT) — Pythonic VTK wrapper; fast path to 3-D
  desktop rendering of meshes and point clouds.
- `vispy` (BSD-3) — GPU-accelerated 2-D/3-D Python library with
  OpenGL backends; good for smoothly-animated particle work.
- `k3d-jupyter` / `ipyvolume` (BSD / MIT) — in-notebook WebGL
  viewers for small-to-mid volumes.
- `datashader` (BSD-3) — server-side aggregation for datasets too
  large to rasterize client-side; pairs with HoloViews.
- `holoviews` + `panel` + `bokeh` (BSD) — interactive notebook /
  dashboard layer rendering to the browser.
- `plotly` / `dash` (MIT) — interactive plotting and dashboard
  framework; WebGL traces for particle work.

**Browser-native rendering.**

- `three.js` (MIT) — canonical WebGL scene graph; basis for most
  interactive science-communication experiences.
- `regl` (MIT) — functional-reactive WebGL wrapper; thinner and
  more hackable than three.js for custom shaders.
- `deck.gl` (MIT) — large-scale point/line/polygon layering;
  natural fit for particle clouds and halo catalogs.
- `WebGPU` — emerging standard shipping in all major browsers,
  with compute shaders suitable for volume raymarching and even
  lightweight in-client re-simulation.
- `CesiumJS` (Apache-2 library, asset licenses vary) — spherical
  geometry and streaming tiles; relevant for CMB or
  celestial-sphere visualizations; asset-licensing footprint
  requires evaluation before adoption.
- `OpenSeadragon` / DeepZoom / Neuroglancer tile servers — tiled
  pyramids for very large 2-D and 3-D fields.

**Streaming and viz-shaped output formats.**

- `Zarr` (MIT) — chunked, cloud- and browser-friendly array store;
  Zarr v3 adds sharding, consolidated metadata, and explicit
  codec conventions.
- `OME-Zarr` — Zarr conventions with multiscale pyramids, heavily
  used in bio-imaging; directly applicable to cosmological and
  radiation-field tile pyramids.
- `glTF` (MIT) — portable 3-D scene format; the right choice for
  any geometry (meshes, instanced particles) shipped to a web
  viewer.
- `Parquet` (Apache-2) — columnar format for particle and halo
  catalogs; streams well to notebook and browser consumers.
- `ADIOS2` — HPC-grade, optimized for high-throughput parallel I/O
  rather than browser consumers; complementary to Zarr, not a substitute.

**Color and typography.**

- `matplotlib` perceptually-uniform maps (`viridis`, `cividis`,
  `inferno`, `magma`) — the scientific baseline.
- `cmasher` (MIT) — extended family of perceptually-uniform,
  color-vision-deficient-safe colormaps aimed at astrophysics.
- `cmocean` (MIT) — perceptual maps tuned for oceanography,
  equally appropriate for astrophysical diverging / cyclic data.
- Cinematic renders typically call for bespoke palettes informed
  by perceptual uniformity rather than the default `jet`-era
  rainbows still common in older codes.

**Unit-aware plotting.**

- `astropy.units` (BSD-3) — the astronomy-standard unit system.
- `unyt` (BSD-3) — lightweight unit system originally split out
  of yt; low-overhead, jittable-friendly, and what the analysis
  code in the ecosystem already carries.
- `pint` (BSD-3) — general-science alternative; adopted mainly
  outside astronomy.

**In-situ and data-as-movies paradigms.**

- `ALPINE / Ascent` (BSD) — in-situ visualization infrastructure
  linked directly into simulation codes; writes Cinema databases
  alongside traditional plotfiles.
- `Cinema` — parameterized image-database framework (Ahrens et
  al.); persists pre-rendered viewpoints so exploratory analysis
  is a database query, not a re-render.
- `ParaView Catalyst` and `VisIt libsim` — historical in-situ
  APIs; relevant mainly as algorithmic references for engines
  that avoid compiled library dependencies.

**Science-communication surfaces.**

- `MyST-NB` + `sphinx-design` — executable narrative documents
  in the engine's docs pipeline; the baseline for theory manual
  content.
- `Jupyter Book` — one level above MyST-NB for book-length
  explainers with cross-references and executable content.
- `Observable` / `Distill.pub` — interactive-article idioms
  (scrollytelling, live widgets, paired math-and-simulation
  panels) that set the bar for educational impact.
- `Streamlit` / `Gradio` — fast-to-author interactive dashboards;
  adopted only where notebook-embedded HoloViews or Panel do not
  suffice.
- `pyodide` / `JupyterLite` — Python in the browser, enabling
  read-execute-explain narratives without a server.

**Visual-regression and testing infrastructure.**

- `pytest-mpl` (BSD) — baseline-image comparison for matplotlib
  figures; the default tool for figure tests.
- `scikit-image` SSIM (BSD) — perceptual image similarity,
  suitable for volume-render and movie diffs where pixel-exact
  comparison is brittle.
- `reg-viz`-style tooling — browser-side visual diffing for the
  web gallery and explainer pages.

## 6.12 Licensing and openness landscape

- **Permissive (BSD / Apache / MIT):**
  Castro, Nyx, MAESTROeX, Quokka, GAMER, Athena++, AthenaK, KHARMA,
  Parthenon, AMReX, Kokkos, Phoebus, GRChombo, thornado, Flash-X,
  Enzo/Enzo-E. Algorithms from these codes can be reproduced with
  attribution and without forcing the rest of the repository into a
  copyleft license.
- **Copyleft (GPL / LGPL / CeCILL):** Arepo, GADGET-4, GIZMO-public, ChaNGa, RAMSES,
  MPI-AMRVAC, BHAC, koral_lite, PLUTO, Dedalus, MESA (LGPL), SWIFT
  (LGPL). Papers and documentation describe the algorithms freely;
  those descriptions are what we work from, and any reimplementation
  must be clean-room.
- **Closed / collaboration-only:** KEPLER, CHIMERA, FORNAX,
  WhiskyTHC, and the private extensions of Arepo (IllustrisTNG,
  Auriga) and GIZMO (FIRE, STARFORGE). Only published papers are
  available.

---

# 8. Verification, Validation, and Specification Methodology

This section surveys the V&V literature across CFD, aerospace, nuclear, and
computational astrophysics, covering the frameworks, standards, and methods
(MMS, PCMM, Richardson extrapolation) and their application — or absence —
in open-source simulation codes.

## 8.1 Foundational V&V Framework

The modern CFD V&V framework was established primarily by two groups:
Roache (independent consultant / Hermosa Publishers) in the 1990s, and
Oberkampf & Trucano at Sandia National Laboratories in the 2000s.

**Patrick J. Roache**, "Quantification of Uncertainty in Computational Fluid
Dynamics," *Annual Review of Fluid Mechanics*, 29:123–160, 1997.
<https://doi.org/10.1146/annurev.fluid.29.1.123>
Established the distinction between verification (did we solve the equations
right?) and validation (did we solve the right equations?), and introduced
the Grid Convergence Index as a standardized convergence reporting method.
This is the paper that gave the field its vocabulary.

**Patrick J. Roache**, *Verification and Validation in Computational Science
and Engineering*, Hermosa Publishers, 1998. ISBN 0-913478-08-3.
The first book-length treatment. Introduced MMS as a first-class code
verification tool.

**William L. Oberkampf and Timothy G. Trucano**, "Verification and Validation
in Computational Fluid Dynamics," *Progress in Aerospace Sciences*,
38:209–272, 2002.
<https://doi.org/10.1016/S0376-0421(02)00005-2>
The most-cited single paper in CFD V&V (~840 citations). Established rigorous
terminology, distinguished code verification from solution verification and
model validation from solution validation, and proposed a taxonomy of error
and uncertainty sources.

**William L. Oberkampf and Christopher J. Roy**, *Verification and Validation
in Scientific Computing*, Cambridge University Press, 2010.
ISBN 978-0-521-11360-1.
<https://doi.org/10.1017/CBO9780511760396>
The definitive textbook: 784 pages covering software engineering for
simulation, MMS, solution verification, model validation, design of
validation experiments, and uncertainty quantification.

## 8.2 Standards Documents

**AIAA G-077-1998**, *Guide for the Verification and Validation of
Computational Fluid Dynamics Simulations*, AIAA, 1998.
<https://arc.aiaa.org/doi/book/10.2514/4.472855>
The first consensus standards document. Defines terminology still in use
today and proposes procedures for each transition in the simulation pipeline.

**ASME VV-20-2009 (reaffirmed 2021)**, *Standard for Verification and
Validation in Computational Fluid Dynamics and Heat Transfer*, ASME.
<https://www.asme.org/codes-standards/find-codes-standards/v-v-20-standard-for-verification-and-validation-in-computational-fluid-dynamics-and-heat-transfer>
The first normative (not just advisory) standard. Introduces the view that
validation is not pass/fail but an assessment of model error at a specific
validation point. Required in aerospace and ASME-regulated industries.

**NASA-STD-7009B** (2024), *Standard for Models and Simulations*, NASA.
<https://standards.nasa.gov/standard/NASA/NASA-STD-7009>
The most operationally demanding regulatory standard. Mandates eight
credibility factors across V&V, operational quality, and supporting evidence
before simulation results may be used to support decisions.

## 8.3 Method of Manufactured Solutions (MMS)

MMS is now the de facto standard first step in code verification for any new
or refactored PDE solver. It verifies that a discretization map converges to
the correct continuous operator at the stated rate, using source terms
computed symbolically from a chosen "manufactured" solution.

**Kambiz Salari and Patrick Knupp**, *Code Verification by the Method of
Manufactured Solutions*, Sandia National Laboratories SAND2000-1444, 2000.
<https://www.osti.gov/biblio/759450>
The canonical reference document. Provides the complete mathematical
procedure, source-term derivation, and application to incompressible and
compressible Navier-Stokes.

**Patrick Knupp and Kambiz Salari**, *Verification of Computer Codes in
Computational Science and Engineering*, Chapman & Hall/CRC, 2003.
ISBN 978-1-58488-264-0.
The book-length treatment. Includes worked examples and discussion of how
MMS identifies coding errors that affect convergence order but not
isolated single-resolution tests.

**Christopher J. Roy**, "Verification of Euler/Navier-Stokes codes using the
method of manufactured solutions," *International Journal for Numerical
Methods in Fluids*, 44:599–620, 2004.
<https://doi.org/10.1002/fld.660>
Demonstrates MMS applied to full compressible flow solvers with detailed
worked examples.

**Limits of MMS.** MMS tests the discretization of interior operators but
does not test the physical model itself, stiff source terms, or problems with
discontinuities. Boundary condition MMS requires additional care. A code that
passes MMS on smooth manufactured data can still fail on physically
realizable initial conditions. Roy (2005) discusses this in detail.

**Christopher J. Roy**, "Review of Code and Solution Verification Procedures
for Computational Simulation," *Journal of Computational Physics*,
205(1):131–156, 2005.
<http://ftp.demec.ufpr.br/CFD/bibliografia/erros_numericos/Roy_2005.pdf>
Comprehensive survey of MMS, Richardson extrapolation, and order-of-accuracy
testing. Widely used as a self-contained technical reference.

## 8.4 Predictive Capability Maturity Model (PCMM)

The PCMM, developed at Sandia, is the closest existing framework to a
formal capability specification discipline.

**William L. Oberkampf, Martin Pilch, and Timothy G. Trucano**, *Predictive
Capability Maturity Model for Computational Modeling and Simulation*, Sandia
National Laboratories SAND2007-5948, 2007.
<https://www.osti.gov/servlets/purl/976951/>
Defines a four-level maturity scale across six elements: geometric/
representation fidelity, physics model fidelity, code verification, solution
verification, model validation, and uncertainty quantification. Originally
developed for nuclear weapons stockpile assessment; now used more broadly.

**The PCMM used prospectively is the closest existing analog to a
capability specification.** The six elements define what evidence must exist
for a simulation capability to be credible at each maturity level. If the
PCMM rubric is filled in *before* the code is written — specifying what
evidence will be produced and at what level — it functions as a formal
capability specification. This use is not described in the literature but is
the natural extension.

## 8.5 Who Has Aimed for This Level of Rigor

The best-practice floor recommended by the Oberkampf & Roy framework — MMS
for code verification, Richardson extrapolation for solution verification,
externally grounded validation experiments — is well understood but rarely
fully applied. The following are the closest examples in the literature.

**FLASH / Flash-X** (see §8.6) applied the CFD V&V framework explicitly and
is the best example from computational astrophysics. The Calder et al. (2002)
paper is the only astrophysics code paper that reads like a V&V document
rather than a capability announcement. The ongoing Dubey et al. work on
verification as a continuous process is also notable. However, the test suite
was built alongside or after the code, not as a prior specification of
correctness claims.

**AMReX-Astro / Castro** (see §8.6) has well-documented ongoing
verification infrastructure: nightly regression tests, manufactured
solution tests, and known-answer convergence problems. The verification
tests are defined by observed code behavior rather than an independent
prior specification of correctness claims.

**Trilinos** (see §8.6) applied the most rigorous software engineering
discipline to scientific computing infrastructure: mandatory unit tests,
continuous integration, defined dependency contracts, and a formal maturity
lifecycle. The rigor is architectural and process-level. Physics correctness
(in the sense of externally grounded claims about what the code computes) is
not within Trilinos's scope.

**NRC thermal-hydraulics codes** (see §8.6) achieved the highest operational
rigor through regulatory requirement: codes cannot be used for nuclear safety
licensing without documented V&V. This is the closest any simulation code
community has come to a mandatory, sustained, externally auditable correctness
discipline. The context is very different (regulatory, industrial, narrow
physical scope) but the principle is the same.

**Goal-oriented error estimation (Oden & Prudhomme 2001).**
*Computers and Mathematics with Applications*, 41(5–6):735–756, 2001.
<https://doi.org/10.1016/S0898122100003175>
Provides a mathematically rigorous path to verified error bounds on specific
output quantities using adjoint-based estimation. This is the most rigorous
approach in the academic literature: the correctness claim ("compute drag
coefficient to 1% accuracy") is stated precisely, and the verification is a
mathematical proof that the code meets it. Uncommon in practice outside
finite-element structural mechanics.

**Summary.** No open-source astrophysics code has applied
externally grounded verification consistently across its physics capabilities
as an ongoing development discipline. FLASH and AMReX-Astro come closest but
are still primarily regression-testing frameworks: they detect drift from a
previously established baseline, which is not the same as verifying against
an external correctness claim. NIST IR 8298 (2020) and a 2025 testing survey
confirm this is the norm across scientific computing, not an astrophysics
peculiarity.

## 8.6 Code Projects with Systematic V&V

**FLASH / Flash-X (University of Chicago / DOE ASCI / Argonne).**
Calder et al. (2002), *ApJS* 143(1):201–229, is one of the first
astrophysics papers to apply the CFD V&V framework explicitly.
<https://doi.org/10.1086/342267>
The FLASH test suite covered known-answer problems, shock physics, and
comparison to laser-driven shock experiments. Dubey et al. (2014) documents
how the component-based architecture enabled ongoing verification across
20+ years of code evolution.
<https://doi.org/10.1177/1094342013505656>

**AMReX-Astro ecosystem (Castro, MAESTROeX, Nyx, Pele).**
<https://amrex-codes.github.io/regression_testing/>
The most systematic ongoing verification infrastructure currently active in
computational astrophysics. Nightly automated regression tests across the
full AMReX ecosystem; Castro's verification page explicitly documents
manufactured solution tests and known-answer problems. This is the current
state of the art in comp-astro V&V.

**Athena / Athena++.**
Stone et al. (2008), *ApJS* 178:137–177. <https://doi.org/10.1086/588755>
Documents a test suite with quantitative convergence measurements for 1D,
2D, and 3D hydro and MHD problems. The test suite was designed for
comparison with other codes and is built into the repository.

**Trilinos (Sandia National Laboratories).**
Heroux et al. (2005), *ACM TOMS* 31(3):397–423.
<https://doi.org/10.1145/1089014.1089021>
Each Trilinos package must meet ASC Software Quality Engineering standards:
unit tests, continuous integration, defined dependency contracts. The
TriBITS lifecycle model formalizes maturity levels from research prototype
to production library. Arguably the best example of software engineering
discipline applied to scientific computing infrastructure.

**NRC thermal-hydraulics codes (TRACE, RELAP).**
<https://www.nrc.gov/about-nrc/regulatory/research/safetycodes>
The most heavily regulated domain: codes cannot be used for nuclear safety
licensing without documented V&V. The NRC's CAMP international
collaboration provides structured validation across ~30 countries.

**SU2 (open-source aerospace CFD).**
<https://su2code.github.io/vandv/MMS_FVM_Navier_Stokes/>
Maintains a publicly documented MMS verification suite for compressible
Navier-Stokes, updated with each code release.

## 8.7 Reproducibility in Scientific Computing

**Ivie and Thain**, "Reproducibility in Scientific Computing," *ACM
Computing Surveys*, 51(3):63, 2018.
<https://doi.org/10.1145/3186266>
Surveys reproducibility barriers across scientific computing. Finds that most
computational experiments are described only informally in papers and that
the code producing the results is rarely available.

**Non-determinism in HPC.** Modern HPC systems produce results that differ
across runs even with identical source code and input, because parallel
floating-point arithmetic is non-associative. This is a hard reproducibility
floor below which bit-reproducibility cannot be guaranteed without explicit
design effort. See Hoefler and Belli (*SC15*, 2015).

**DOE Correctness for Scientific Computing (CS2) Initiative, 2025.**
<https://computing.llnl.gov/projects/formal-methods-correctness>
A 2025 DOE/NSF joint program funding research into formal methods and
correctness for scientific computing codes. Signals that the gap between
software engineering rigor and simulation practice is recognized at the
funding-agency level.
