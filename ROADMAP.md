# Cosmic Foundry — Implementation Roadmap

This document is the high-level, time-ordered plan for building the
Cosmic Foundry engine. It is derived from RESEARCH.md — specifically
the union of capabilities in §6 and the engine-design implications in
§7 — and expresses a development sequence for a **fully self-contained
Python engine** whose performance is delivered by runtime code
generation rather than by a compiled C++ core.

The roadmap is intentionally strategic: each epoch is weeks-to-months
of real work, not a sprint. Granularity is "order of operations,"
not "task list." The expectation is that finer-grained plans will be
written per epoch as its entry criteria are met.

---

## 1. Framing

**Goal.** A self-contained engine whose capability set is the union of
capabilities catalogued in RESEARCH.md §6 — covering uniform and
adaptive meshes, Newtonian through numerical-relativistic
(magneto)hydrodynamics, radiation transport, microphysics, gravity,
N-body, cosmology, particle / meshless / moving-mesh methods, and
stellar evolution.

**Principle — infrastructure first, physics second.** Every physics
module should land as a paper-scale effort on top of mature
primitives. Infrastructure mistakes in Epoch 1–2 cost an order of
magnitude more to fix than in any later epoch.

**Principle — easy case before hard case.** Within each subsystem,
ship the simplest instance before the general one: uniform grid
before AMR; Newtonian before special-relativistic before GR before
dynamical-spacetime; gray flux-limited diffusion before multigroup
M1; SPH before moving mesh.

**Principle — autodiff as a first-class tool.** JAX and SymPy
autodiff eliminate hand-derived Jacobians for stiff integrators,
primitive-variable recovery, BSSN/Z4c source terms, and parameter
sensitivity. Design physics modules so their Jacobians are obtained
from autodiff unless there is a demonstrated performance reason to
hand-code them.

**Principle — licensing discipline.** Capabilities with a
permissive-licensed reference code (RESEARCH.md §6.12) are studied
and adapted with attribution. Copyleft references are consulted
through published papers only; any reimplementation is clean-room.
Closed / collaboration-only codes (KEPLER, CHIMERA, FORNAX,
WhiskyTHC, private Arepo/GIZMO modules) contribute only through
their publications.

---

## 2. Technology baseline

These are initial commitments — revisitable, but changes should
require an ADR.

- **Python ≥3.11** as the single source language. No compiled
  extensions shipped from this repository; any native code is
  produced at runtime by a codegen backend.
- **JAX + XLA** as the primary array, autodiff, and JIT layer.
  `pjit` / `shard_map` provide device- and host-parallelism on top
  of MPI.
- **Secondary kernel backends**, wrapped behind a `@kernel`
  descriptor layer so they are interchangeable per-kernel. These
  backends are **designed for but not exercised in the early
  epochs** — only JAX is brought up first, with the others stubbed
  so that additional adapters can slot in later without surgery:
  - **Numba** — CPU SIMD and CUDA kernels where XLA's shape /
    control-flow constraints are limiting.
  - **Taichi** — particle / SPH / unstructured / moving-mesh
    workloads; mature simulation DSL.
  - **NVIDIA Warp** — GPU particle, tree, and neighbor kernels;
    Monte Carlo transport.
  - **Triton** — hand-tuned GPU kernels for the few cases that need
    full control.
- **mpi4py** for message passing; `mpi4jax` and `jax.distributed` for
  JAX interop.
- **h5py + parallel HDF5** for checkpoints and plotfiles. ADIOS2
  (Python) considered later.
- **NumPy + SciPy** for low-cost CPU work and reference
  implementations.
- **SymPy** for symbolic derivation of fluxes, Jacobians, and
  curvature tensors, with codegen into JAX or Numba.
- **pybind11 / ctypes** tolerated only as emergency escape hatches.
- **pytest + hypothesis** for testing; golden-file regression via
  HDF5 snapshot comparison.
- **Sphinx + MyST-NB** for documentation (already provisioned by
  miniforge).
- **PEP 621 `pyproject.toml`** with optional extras for each kernel
  backend, so users can install a minimum stack for their hardware.

---

## 3. Epochs

### Epoch 0 — Project bootstrap

Epoch 0 delivers the project scaffolding every later epoch
assumes: a Python package, a tooling and CI stack, a documentation
system, an architecture-decision-record process, and a trivial
end-to-end "hello" entrypoint that proves the environment works.

The scope is deliberately narrow. Only the **JAX** kernel backend
is exercised here; Numba, Taichi, Warp, and Triton are listed as
optional extras in `pyproject.toml` and accommodated in the
descriptor-layer design, but no adapters are written, installed,
or tested in Epoch 0. This keeps the bootstrap small and avoids
committing to the details of the kernel interface before Epoch 1
has a chance to evolve it against a real workload.

#### 0.1 Repository layout

```
cosmic-foundry/
├── cosmic_foundry/                 # main Python package
│   ├── __init__.py                 # version, top-level API
│   ├── cli/                        # click-based entry points
│   ├── kernels/                    # kernel descriptor + JAX adapter (stub)
│   ├── mesh/                       # mesh primitives (stub)
│   ├── io/                         # HDF5 + plotfile helpers (stub)
│   ├── physics/                    # physics module namespace (stub)
│   └── _version.py                 # populated by setuptools-scm or hatch-vcs
├── tests/                          # pytest tree, mirrors the package layout
├── docs/                           # Sphinx sources (MyST-NB enabled)
├── examples/                       # runnable example scripts and notebooks
├── benchmarks/                     # perf harnesses (wired up in Epoch 1)
├── adr/                            # architecture decision records
├── .github/workflows/              # CI definitions
├── environment/                    # miniforge setup (already present)
├── scripts/                        # developer scripts (already present)
├── assets/                         # brand assets (already present)
├── pyproject.toml                  # PEP 621 metadata and tooling config
├── ruff.toml                       # lint rules (or section in pyproject)
├── .pre-commit-config.yaml
├── .editorconfig
├── README.md                       # expanded from the current one-liner
├── CONTRIBUTING.md
├── AI.md                           # already present
├── CLAUDE.md / CODEX.md / GEMINI.md
├── RESEARCH.md                     # already present
├── ROADMAP.md                      # this document
└── LICENSE
```

Only `cosmic_foundry/`, `tests/`, `docs/`, `examples/`,
`benchmarks/`, `adr/`, and `.github/workflows/` are genuinely new
in Epoch 0; everything else is scaffolding files at the repo root
or extensions of existing directories.

#### 0.2 Packaging — `pyproject.toml`

- **Build backend:** `hatchling` with `hatch-vcs` for version
  derivation from git tags.
- **Core runtime dependencies:** `numpy`, `scipy`, `jax`, `jaxlib`,
  `h5py`, `mpi4py`, `click`, `sympy`, `typing-extensions`.
- **Optional extras (stubbed, not validated in Epoch 0):**
  - `dev`: pytest, pytest-cov, pytest-mpi, pre-commit, black, ruff,
    mypy, hypothesis.
  - `docs`: sphinx, myst-nb, furo, sphinx-design, sphinx-autodoc2.
  - `numba`, `taichi`, `warp`, `triton`: each pins the relevant
    package at a known-good version. These extras exist so the
    descriptor layer has a clear target, but they are not
    installed or imported by default code paths.
- **Console scripts:** `cosmic-foundry = cosmic_foundry.cli:main`.
- **Python:** `requires-python = ">=3.11"` to match miniforge.

#### 0.3 Tooling and code quality

- **Formatting:** black (default settings, line length 88).
- **Lint:** ruff with a curated rule set (pycodestyle, pyflakes,
  isort, bugbear, pyupgrade, numpy-specific rules). Configuration
  lives in `pyproject.toml`.
- **Type-checking:** mypy in strict mode for `cosmic_foundry/` and
  non-strict for `tests/`. JAX stubs pulled in as needed.
- **Pre-commit:** black, ruff, mypy, end-of-file fixer,
  trailing-whitespace, check-yaml, check-toml. Installed on first
  developer clone via `pre-commit install` documented in
  CONTRIBUTING.md.
- **Editor config:** `.editorconfig` for tab/space consistency.

#### 0.4 Continuous integration

A single GitHub Actions workflow, `.github/workflows/ci.yml`,
running on push and pull-request:

- **OS matrix:** Linux only (ubuntu-latest).
- **Python matrix:** 3.11 only.
- **Steps:**
  1. Check out, set up miniforge via the project's
     `environment/setup_environment.sh`, cache the resulting
     environment keyed on `environment/*.yml`.
  2. `pip install -e .[dev,docs]`.
  3. `pre-commit run --all-files`.
  4. `mypy cosmic_foundry`.
  5. `pytest -q` (single-rank CPU tests).
  6. `sphinx-build -W docs docs/_build/html` (fail on warnings).
- A second workflow, `gpu.yml`, is scaffolded with
  `workflow_dispatch: {}` only — it contains placeholder steps for
  GPU runs but is not wired to any runner. Enabling it is deferred
  to whenever GPU runners become available.
- Multi-rank MPI tests (`pytest-mpi`) are scaffolded in the tests
  tree but gated behind an `--mpi` marker and not run in CI in
  Epoch 0; they will be turned on in Epoch 1.

#### 0.5 Documentation scaffolding

- **Engine:** Sphinx with `myst-nb`, `furo` theme, `sphinx-design`.
- **Pages seeded:**
  - `index.md` — overview + links to RESEARCH.md, ROADMAP.md,
    ADR index.
  - `getting-started.md` — environment setup, install, running
    `cosmic-foundry hello`.
  - `contributing.md` — linked from `CONTRIBUTING.md`.
  - `coding-standards.md` — style, docstrings (NumPy format),
    typing expectations, test philosophy.
  - `theory/` — empty section with a stub page per physics module
    planned in later epochs, each saying "to be written."
  - `api/` — autodoc entry point, populated as modules land.
  - `adr/index.md` — generated listing of ADRs.
- **Build:** local `sphinx-build` only. RTD / Pages publishing is
  deferred until there is real content to host.

#### 0.6 Architecture Decision Record process

An `adr/` directory with `adr-template.md` and five seed ADRs
codifying commitments already documented elsewhere:

- **ADR-0001** — Python-only engine with runtime code generation
  (references RESEARCH.md §7 and the user decisions recorded in the
  ROADMAP planning notes).
- **ADR-0002** — JAX + XLA as the primary kernel backend; Numba,
  Taichi, Warp, and Triton accommodated in the descriptor layer
  but deferred.
- **ADR-0003** — MPI (via mpi4py and `jax.distributed`) is in the
  baseline from Epoch 1.
- **ADR-0004** — Documentation is authored in Sphinx + MyST-NB and
  versioned alongside code.
- **ADR-0005** — Branch and PR discipline, single-file
  documentation-commit exception, and attribution rules (mirrors
  AI.md, made explicit for new contributors).

Each ADR follows the same short format: context, decision, status,
consequences. Future epochs open new ADRs rather than mutate old
ones.

#### 0.7 `cosmic-foundry hello` entry point

A minimal CLI exercise that proves the toolchain is wired up:

- Parse no arguments (Epoch 0) beyond `--help` / `--version`.
- Initialize MPI via `mpi4py` and report `rank` / `size`.
- Query JAX device list and print backend name + device summary
  from rank 0.
- Run a trivially small JAX `jit` (e.g. a 32³ Laplacian smoke
  test) on rank 0 to confirm the JIT path is functional.
- Exit cleanly with code 0, and with a non-zero code plus
  actionable message if any step fails.

This is the one runtime behavior Epoch 0 adds; it also serves as
the first integration test (invoked via `subprocess` in pytest).

#### 0.8 README and CONTRIBUTING

- **README.md** expands to: one-paragraph positioning, a "quick
  start" block (setup_environment.sh → activate → `pip install
  -e .[dev]` → `cosmic-foundry hello`), pointers to RESEARCH.md
  and ROADMAP.md, license pointer.
- **CONTRIBUTING.md** captures the developer workflow: fork,
  branch naming, 100-line-per-commit guideline from AI.md with
  the documented documentation-commit exception, PR expectations,
  pre-commit hook installation, and the ADR process for
  cross-cutting decisions.

#### 0.9 Visualization scaffolding

Even though no physics runs in Epoch 0, the house-style
commitments that downstream visualization depends on are made
here so they cannot drift in later epochs:

- `cmasher`, `cmocean`, and matplotlib perceptually-uniform maps
  pinned in the `docs` extra; `unyt` pinned in the core runtime
  dependency list for unit-aware plotting.
- `docs/gallery/` stub page using the `sphinx-design` card layout;
  each later physics epoch populates a card.
- `tests/visual/` subtree with a trivial `pytest-mpl`
  baseline-image case (e.g. a sinusoid rendered under the house
  colormap) wired into CI, so later visual-regression work plugs
  into an already-green harness.
- Accessibility and performance budget stub (`docs/accessibility
  .md`) naming the targets the public gallery will enforce: WCAG
  2.2 AA contrast, colourblind-safe palettes, alt text, mobile
  LCP under 2.5 s on a 4G profile, bytes-on-wire budgets per
  tiled dataset.
- ADR-0006 (visualization and science-communication stack) is
  referenced from the ADR index; its decisions constrain the
  Epoch 2 I/O design and the dedicated Epoch 3 viewer work.

No runtime rendering code is written in Epoch 0; the scaffolding
is purely dependencies, stubs, and the visual-regression harness.

#### 0.10 Exit criteria

Epoch 0 is complete when all of the following hold:

- CI is green on `main` with lint, type-check, test, and docs
  build all passing.
- A developer on a fresh Linux machine can, in under ten minutes,
  clone the repo, run `bash environment/setup_environment.sh`,
  activate the env, `pip install -e .[dev]`, run `pytest`, and
  run `cosmic-foundry hello` with each step succeeding.
- `pre-commit run --all-files` is clean on every committed file.
- `sphinx-build -W docs docs/_build/html` builds without errors or
  warnings.
- The five seed ADRs are merged and the `adr/` index renders in
  the docs; ADR-0006 (visualization stack) is also merged.
- No Numba / Taichi / Warp / Triton code is imported by any
  default code path; those extras install but remain unexercised.
- `pytest-mpl` runs green against the baseline-image case in
  `tests/visual/` on the pinned house colormap.

#### 0.11 Explicitly deferred to later epochs

- The kernel descriptor itself (interface, semantics, backend
  dispatch) is *sketched* in `cosmic_foundry/kernels/__init__.py`
  as a placeholder only. The real design lands in Epoch 1.
- Non-JAX backend adapters (Numba, Taichi, Warp, Triton).
- GPU CI.
- Multi-rank MPI CI.
- Documentation publishing (RTD, Pages).
- Benchmark harness in `benchmarks/` beyond an empty directory.
- Problem-setup DSL vs YAML vs Python-API decision — this is
  first-touched in Epoch 4 (Newtonian hydrodynamics) at the
  earliest.

### Epoch 1 — Kernel abstraction and multi-backend core

The one piece of infrastructure we genuinely have to invent:

- A `Kernel` descriptor — a declarative spec of a stencil or
  element-wise operation over typed arrays with named axes — plus
  adapters that lower it to JAX, Numba, Taichi, Warp, or Triton at
  runtime.
- A `ShardedField` distributed-array primitive built on
  `jax.distributed` + mpi4py for cross-device / cross-host cases.
- Parallel HDF5 I/O via h5py+MPI.
- Deterministic structured logging and error handling.

**Exit criterion:** a 3-D 7-point Laplacian benchmark runs under
every backend at documented roofline fractions; a multi-rank
correctness test proves backend and sharding equivalence; and a
reference render of one benchmark slice under the house colormap
is committed as the first production visual-regression artifact.

### Epoch 2 — Mesh and AMR

Bring up the mesh hierarchy the physics modules will live on:

- Uniform structured grid with ghost-cell exchange and domain
  decomposition.
- Block-structured AMR hierarchy expressed as JAX pytrees of blocks
  with cell / face / edge / node centering, subcycling in time, and
  refinement-flux correction.
- Task-graph driver for asynchronous dependency scheduling across
  ranks and devices.
- Plotfile writer and yt-compatible metadata.
- Zarr v3 writer with an OME-Zarr-style multiscale pyramid
  convention emitted alongside the plotfile, so every AMR output
  is simultaneously HPC-analysis-ready (HDF5 / plotfile) and
  browser-streamable (Zarr). Pinned by ADR-0006.

**Exit criterion:** a second-order advection test converges at
design order on AMR, runs identically on CPU and GPU, produces
rank-invariant output, and writes a matching Zarr pyramid that
the Epoch 3 viewer MVP consumes without hand editing.

### Epoch 3 — Visualization and science communication

The visualization stack that every physics epoch will feed into —
and the public-facing science-communication surface. Promoted out
of the former Stretch Epoch 12 (subgrid physics and observables,
now Stretch Epoch 13) because excellence at visualization is a core
engine requirement, and because rendering-layer and output-format
choices must precede the first physics module to avoid expensive
rewrites. Builds directly on the Zarr writer delivered in Epoch 2
and the house-style scaffolding from Epoch 0.

- Unit-aware plotting layer over `unyt`, bridging `astropy.units`
  at the engine boundary.
- In-engine rendering primitives in JAX — camera, 2-D slice
  sampler, 3-D volume raymarcher, particle projector — shared
  between batch CPU / GPU renders and the browser viewer so the
  same kernels drive both.
- WebGPU viewer package (with WebGL2 fallback) that consumes
  Zarr tile pyramids and glTF geometry. Shaders authored once in
  WGSL and transpiled for WebGL2.
- MyST-NB explainer template embedding the viewer as an
  interactive widget; `sphinx-design` gallery page published as
  part of the docs site.
- Visual-regression harness upgraded from the Epoch 0 stub:
  `pytest-mpl` for figures, an SSIM-based diff harness for
  renders and short movies, references in Git LFS.
- Accessibility and performance budgets from ADR-0006
  (WCAG 2.2 AA, colourblind-safe palettes, mobile LCP < 2.5 s,
  per-dataset bytes-on-wire) codified as CI checks against the
  public gallery build.
- Public gallery (GitHub Pages) seeded with the Epoch 2 AMR
  advection test as its first interactive entry; later physics
  epochs each add one canonical live demo.

**Exit criterion:** the Epoch 2 AMR advection test renders as a
live WebGPU page on the public docs site from checkpoint data
with no hand editing; the page meets the accessibility and
performance budgets under CI; and the visual-regression harness
is green on figure, render, and movie references.

### Epoch 4 — Newtonian hydrodynamics

The first physics module and the template for every subsequent
one:

- Finite-volume Godunov with PPM reconstruction (SymPy-derived
  stencils).
- Riemann solvers: HLLC, HLLE, Roe.
- CFL time-stepping and passive scalars.
- Golden regression suite: Sod, Sedov, Noh, blast wave,
  Kelvin–Helmholtz, Rayleigh–Taylor.

**Exit criterion:** the standard hydro test battery matches
reference solutions across all kernel backends, and a Sod
shock-tube explainer page (live slider over γ and initial
conditions, unit-labelled axes, perceptual colormap) ships to
the public gallery as the first physics demo.

### Epoch 5 — Self-gravity and N-body

Close the loop with gravitational dynamics:

- Geometric multigrid Poisson on the AMR hierarchy (JAX where
  possible; Taichi fallback for irregular smoothers).
- Particle infrastructure — cell-in-cloud deposition, neighbor
  search — on Warp or Taichi.
- Barnes–Hut tree gravity with Ewald summation for periodic BCs.
- FMM prototype.

**Exit criterion:** Zel'dovich pancake and hydrostatic-equilibrium
tests match reference solutions, and a Zel'dovich-pancake live
explainer ships to the gallery with linked phase-space and
density-field views.

### Epoch 6 — Magnetohydrodynamics and non-ideal transport

Extend the hydro module into the MHD regime:

- Constrained-transport ideal MHD — flux-CT and staggered vector-
  potential formulations.
- Super-time-stepping for parabolic terms.
- Resistive, Hall, and ambipolar MHD.
- Anisotropic thermal conduction and viscosity.

**Exit criterion:** Orszag–Tang, MHD rotor, and MHD blast wave
match literature benchmarks, and an Orszag–Tang live explainer
ships to the gallery with field-line overlays and a resistivity
slider.

### Epoch 7 — Microphysics sub-layer

Bring up the equations of state and reaction networks that later
physics modules depend on:

- Abstract EOS interface with ideal-gas, Helmholtz, piecewise
  polytropic, and tabulated nuclear finite-T implementations.
  Tables are JAX-jittable piecewise interpolants.
- Reaction-network engine with autodiff-generated Jacobians,
  α-network reference, and a path to large adaptive networks.
- Primordial and metal cooling tables.
- Radiation opacities.

**Exit criterion:** thermonuclear flame and primordial cooling
benchmarks match published results, and a 1-D thermonuclear-flame
explainer ships with interactive EOS / network switching.

### Epoch 8 — Radiation transport

Close the radiation-hydrodynamics loop:

- Gray flux-limited diffusion.
- Multigroup FLD.
- Two-moment M1 with reduced speed of light.
- Short-characteristics variant.
- Interfaces left open for DG and Monte Carlo (the latter is a
  natural fit for Warp).

**Exit criterion:** shadow, linear-wave, and radiating-shock
tests pass, and a radiating-shock explainer ships with a
reduced-speed-of-light slider and linked spectral / spatial
views.

### Epoch 9 — Relativistic physics

The hardest grid-based physics target:

- Special-relativistic hydrodynamics.
- GR hydro and GRMHD on fixed stationary spacetimes, with robust
  primitive-variable recovery (multiple schemes, autodiff
  Jacobians).
- Dynamical-spacetime evolution via BSSN or Z4c with moving-
  puncture gauge. SymPy derives the Einstein source terms.

**Exit criterion:** Fishbone–Moncrief torus and binary-black-hole
inspiral benchmarks pass, and a binary-black-hole inspiral
cinematic ships to the gallery — orbit-then-merger flythrough,
gravitational-wave strain panel, unit-labelled axes — as the
flagship demo of the engine's relativistic capability.

### Epoch 10 — Particle / meshless hydrodynamics and cosmology

The second mesh paradigm and the cosmological stack:

- SPH with modern pressure–energy and density-independent
  variants; neighbor loops on Taichi or Warp.
- Meshless finite-mass and finite-volume methods.
- Comoving FRW integrator.
- 2LPT initial conditions.
- On-the-fly FOF and SUBFIND-style halo finders.
- Light-cones and high-dynamic-range power-spectrum estimator.

**Exit criterion:** a cosmological box reproduces a published
reference at fixed resolution within documented tolerance, and a
cosmological-flythrough explainer ships to the gallery with a
light-cone tour and a live halo-mass-function panel.

### Stretch Epoch 11 — Moving Voronoi mesh

The Arepo-class replication target. Deferred until the above are
stable so the engine can absorb the complexity. Likely implemented
on Taichi or Warp with SciPy-based CPU fallbacks for correctness
tests.

### Stretch Epoch 12 — Stellar evolution

1-D Lagrangian stellar structure with an implicit solver (JAX
autodiff for the block-tridiagonal Jacobian), adaptive mesh and
timestep, coupled nuclear burning, rotation, and mixing; a binary
evolution interface compatible with the multi-D explosive modules
so that progenitor states flow naturally into merger and supernova
runs.

### Stretch Epoch 13 — Subgrid physics and observables

- Plugin interface for cooling / star formation / stellar and SN
  feedback / AGN seeding and feedback / chemical enrichment, so
  that EAGLE-, COLIBRE-, and FIRE-class recipes can be expressed
  inside the engine.
- On-the-fly synthetic EUV, X-ray, and spectral-line observables.
- In-situ rendering passes wired into the Epoch 3 pipeline so
  long-running simulations emit gallery-ready assets during
  runtime rather than only at checkpoint.

---

## 4. Cross-cutting, continuous concerns

These grow every epoch; they are not tied to a single phase.

- **Testing.** Each physics epoch adds regression problems locked
  behind golden HDF5 outputs. Tests run across every supported
  kernel backend.
- **Performance.** Roofline analysis and strong / weak scaling
  tracked from Epoch 1 onward. Per-backend numbers are published so
  backend choice is empirical rather than ideological.
- **Documentation.** User guide, theory manual, and API reference
  advance alongside code. MyST-NB notebooks ship as part of docs.
- **Attribution.** Capabilities derived from a permissive-licensed
  reference cite that code and its paper at the algorithmic level.
  Copyleft references are cited but never copied.
- **Reproducibility.** Bitwise-reproducible runs are a goal for
  single-backend CPU execution (following GAMER-2, GADGET-4); full
  cross-backend bitwise reproducibility is not expected.
- **Visualization and communication.** Excellence at visualization
  is a core engine requirement, not a downstream concern. Every
  physics epoch from Epoch 4 onward ships a canonical live demo
  (interactive WebGPU explainer or notebook widget, perceptual
  colormaps, unit-labelled axes) alongside its regression
  benchmarks. The public gallery is regenerated per release with
  accessibility (WCAG 2.2 AA, colourblind-safe palettes, alt text)
  and mobile performance budgets (LCP < 2.5 s on a 4G profile,
  per-dataset bytes-on-wire targets) enforced in CI. Visual-
  regression references (figures, renders, short movies) live in
  Git LFS and are updated only by explicit tag-and-review, never
  auto-replaced. See ADR-0006 for the stack choices.

---

## 5. Crossroads / open decisions

These are flagged now and must be resolved before entering the
named epoch, via an ADR:

- **Host-level parallelism model** — JAX `pjit` / `shard_map` vs
  explicit mpi4py, or both. Likely both, with the former inside a
  node and the latter between nodes. Decide before Epoch 1 stable.
- **In-engine renderer vs. yt hand-off** — whether the camera,
  slice sampler, and volume raymarcher live inside the engine
  (and are therefore shareable between CPU batch, GPU batch, and
  in-browser WebGPU) or are delegated to yt + widgyts. ADR-0006
  currently proposes in-engine; confirm before Epoch 3 starts.
- **Web streaming format — Zarr vs. ADIOS2** — the canonical
  output consumed by browser viewers. Expected to be Zarr v3 for
  its mature browser story with ADIOS2 retained for HPC analysis.
  Decide before Epoch 3.
- **Browser rendering target — WebGPU-first vs. WebGL2-first** —
  whether engine-side shaders are authored against WebGPU with a
  WebGL2 fallback, or vice versa. Decide before Epoch 3.
- **Scene-graph / camera ownership** — whether the engine ships
  its own minimal scene abstraction (transforms, cameras, lights)
  or adopts one from `three.js` / `pyvista`. Decide before
  Epoch 3.
- **Preferred particle / SPH backend** — Taichi vs NVIDIA Warp.
  Decide before Epoch 5.
- **Symbolic vs hand-written microphysics** — how much of the EOS
  and reaction-network machinery is SymPy-derived. Decide before
  Epoch 7.
- **NR discretization** — whether dynamical-spacetime evolution in
  Epoch 9 stays on block AMR or adopts a DG / spectral
  discretization. Decide before Epoch 9.
- **Problem-setup surface** — pure Python API, YAML + validated
  schema, or a small domain-specific language. Decide before the
  first external user.
- **Compiled-extension exception** — when, if ever, to relax the
  "no compiled extensions" rule if a single kernel refuses to hit
  performance targets under any backend. Revisit whenever the
  question is forced by data.

---

## 6. Relationship to RESEARCH.md

Every epoch above traces back to specific capabilities in
RESEARCH.md §6:

| Epoch | RESEARCH.md §6 capabilities covered |
|------:|-------------------------------------|
| 0 | — (scaffolding; viz house style seeded from §6.11) |
| 1 | §6.9 (parallelism), §6.10 (I/O), §6.8 (solver infrastructure) |
| 2 | §6.1 (meshes and AMR), §6.10 (plotfiles + Zarr) |
| 3 | §6.11 (visualization and science communication), §6.10 (web streaming) |
| 4 | §6.2 (Newtonian hydro) |
| 5 | §6.4 (gravity and N-body) |
| 6 | §6.2 (MHD, non-ideal) |
| 7 | §6.5 (EOS and microphysics), §6.3 (cooling) |
| 8 | §6.3 (radiation transport) |
| 9 | §6.2 (SR / GR / NR), §6.8 (primitive recovery) |
| 10 | §6.1 (SPH / meshless), §6.4 (cosmology) |
| 11 | §6.1 (moving mesh) |
| 12 | §6.7 (stellar evolution) |
| 13 | §6.6 (subgrid), §6.10 (diagnostics) |

No epoch covers a capability that is not named in RESEARCH.md §6,
and every capability named in §6 appears in at least one epoch.
