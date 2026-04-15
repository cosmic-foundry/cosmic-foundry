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

The implication for Cosmic Foundry: task sequencing (latency hiding)
and domain batching (kernel-launch amortization) are separate
responsibilities that should live in separate layers. See
`roadmap/epoch-01-kernels.md` §Design constraints.

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

The direct analogy in this engine is SymPy-based Jacobian generation
(ADR-0001): the same principle — derive the Jacobian analytically
from the symbolic rate expressions rather than differencing the RHS —
applies equally to nuclear reaction networks.

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
