# Epoch 0 — Project bootstrap

> Part of the [Cosmic Foundry roadmap](index.md).

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

## 0.1 Repository layout

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

## 0.2 Packaging — `pyproject.toml`

- **Build backend:** `hatchling` with `hatch-vcs` for version
  derivation from git tags.
- **Core runtime dependencies:** `numpy`, `scipy`, `jax`, `jaxlib`,
  `h5py`, `click`, `sympy`, `typing-extensions`. (`mpi4py` is *not*
  in the core set — ADR-0003 places MPI behind the optional `mpi`
  extra below.)
- **Optional extras (stubbed, not validated in Epoch 0):**
  - `dev`: pytest, pytest-cov, pre-commit, black, ruff, mypy,
    hypothesis. (`pytest-mpi` is not in `dev`; it moves to the
    optional `mpi` extra below alongside `mpi4py` / `mpi4jax`,
    since ADR-0003 relegates MPI to an optional per-site fallback.)
  - `mpi` *(optional, not installed by default)*: `mpi4py`,
    `mpi4jax`, `pytest-mpi`. For sites where `jax.distributed`
    cannot initialize over the native interconnect (ADR-0003).
  - `docs`: sphinx, myst-nb, furo, sphinx-design, sphinx-autodoc2.
  - `numba`, `taichi`, `warp`, `triton`: each pins the relevant
    package at a known-good version. These extras exist so the
    descriptor layer has a clear target, but they are not
    installed or imported by default code paths.
- **Console scripts:** `cosmic-foundry = cosmic_foundry.cli:main`.
- **Python:** `requires-python = ">=3.11"` to match miniforge.

## 0.3 Tooling and code quality

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

## 0.4 Continuous integration

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
- Multi-host `jax.distributed` tests are scaffolded in the tests
  tree behind a `--multihost` marker (per ADR-0003) and not run in
  CI in Epoch 0; they will be turned on in Epoch 1 once a Field
  placement smoke test exists on a two-process `jax.distributed`
  harness.

## 0.5 Documentation scaffolding

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

## 0.6 Architecture Decision Record process

An `adr/` directory with `adr-template.md` and five seed ADRs
codifying commitments already documented elsewhere:

- **ADR-0001** — Python-only engine with runtime code generation
  (references RESEARCH.md §7 and the user decisions recorded in the
  ROADMAP planning notes).
- **ADR-0002** — JAX + XLA as the primary kernel backend; Numba,
  Taichi, Warp, and Triton accommodated in the descriptor layer
  but deferred.
- **ADR-0003** — `jax.distributed` + NCCL / GLOO as the
  host-parallelism baseline from Epoch 1; MPI is an optional
  per-site fallback, not in the baseline.
- **ADR-0004** — Documentation is authored in Sphinx + MyST-NB and
  versioned alongside code.
- **ADR-0005** — Branch and PR discipline, single-file
  documentation-commit exception, and attribution rules (mirrors
  AI.md, made explicit for new contributors).

Each ADR follows the same short format: context, decision, status,
consequences. Future epochs open new ADRs rather than mutate old
ones.

## 0.7 `cosmic-foundry hello` entry point

A minimal CLI exercise that proves the toolchain is wired up:

- Parse no arguments (Epoch 0) beyond `--help` / `--version`.
- Call `jax.distributed.initialize()` (ADR-0003) and report
  `process_index` / `process_count` along with the local / global
  device lists.
- Print the JAX backend name and device summary from
  `process_index == 0`.
- Run a trivially small JAX `jit` (e.g. a 32³ Laplacian smoke
  test) on `process_index == 0` to confirm the JIT path is
  functional.
- Exit cleanly with code 0, and with a non-zero code plus
  actionable message if any step fails.

This is the one runtime behavior Epoch 0 adds; it also serves as
the first integration test (invoked via `subprocess` in pytest).

## 0.8 README and CONTRIBUTING

- **README.md** expands to: one-paragraph positioning, a "quick
  start" block (setup_environment.sh → activate → `pip install
  -e .[dev]` → `cosmic-foundry hello`), pointers to RESEARCH.md
  and ROADMAP.md, license pointer.
- **CONTRIBUTING.md** captures the developer workflow: fork,
  branch naming, 100-line-per-commit guideline from AI.md with
  the documented documentation-commit exception, PR expectations,
  pre-commit hook installation, and the ADR process for
  cross-cutting decisions.

## 0.9 Visualization scaffolding

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
  2.2 AA contrast, colorblind-safe palettes, alt text, mobile
  LCP under 2.5 s on a 4G profile, bytes-on-wire budgets per
  tiled dataset.
- ADR-0006 (visualization and science-communication stack) is
  referenced from the ADR index; its decisions constrain the
  Epoch 2 I/O design and the dedicated Epoch 4 viewer work.

No runtime rendering code is written in Epoch 0; the scaffolding
is purely dependencies, stubs, and the visual-regression harness.

## 0.10 Exit criteria

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

## 0.11 Explicitly deferred to later epochs

- The kernel descriptor itself (interface, semantics, backend
  dispatch) is *sketched* in `cosmic_foundry/kernels/__init__.py`
  as a placeholder only. The real design lands in Epoch 1.
- Non-JAX backend adapters (Numba, Taichi, Warp, Triton).
- GPU CI.
- Multi-host `jax.distributed` CI (turned on in Epoch 1; see §0.4).
- Documentation publishing (RTD, Pages).
- Benchmark harness in `benchmarks/` beyond an empty directory.
- Problem-setup DSL vs YAML vs Python-API decision — this is
  first-touched in Epoch 4 (Newtonian hydrodynamics) at the
  earliest.
