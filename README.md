# Cosmic Foundry

Cosmic Foundry is a general-purpose PDE simulation engine, optimized
for astrophysical use cases. The foundational commitments governing its
design are in [**Architectural basis**](ARCHITECTURE.md#architectural-basis).

## Quick start

```bash
# Clone the fork
git clone git@github.com:<your-fork>/cosmic-foundry.git
cd cosmic-foundry

# One-time setup (~5 min)
bash scripts/setup_environment.sh

# Start an agent session (activates environment automatically)
./scripts/start_agent.sh claude   # or gemini / codex

# Or activate manually and work directly
source scripts/activate_environment.sh
cosmic-foundry hello
```

`cosmic-foundry hello` prints the JAX backend, device list, and a JIT
smoke-test result. If it exits 0, the environment is correctly wired.

## Documentation

Build the docs locally:

```bash
sphinx-build -W docs docs/_build/html
```

## Repository layout

Start here:

| Directory | What it is | Read first |
|---|---|---|
| `cosmic_foundry/theory/` | Pure mathematical ABCs — sets, manifolds, discretizations, functions, fields. No JAX dependency. | `theory/__init__.py` |
| `cosmic_foundry/computation/` | Distance-1 concrete implementations of theory ABCs. JAX-backed. | `computation/__init__.py` |
| `cosmic_foundry/geometry/` | Concrete manifolds and simulation domains: `EuclideanSpace`, `MinkowskiSpace`, `Domain`. | `geometry/__init__.py` |
| `cosmic_foundry/mesh/` | Spatial partitioning: uniform Cartesian patches, domain partition, halo fill. | `mesh/__init__.py` |
| `cosmic_foundry/io/` | Array I/O — write to HDF5, merge rank files. | `io/__init__.py` |
| `cosmic_foundry/observability/` | Structured logging. | `observability/__init__.py` |
| `cosmic_foundry/manifests/` | Manifest infrastructure — HTTP client, schema validation, provenance. | `manifests/__init__.py` |
| `cosmic_foundry/cli/` | CLI entry point (`cosmic-foundry`). | `cli/main.py` |
| `tests/` | Test suite. `tests/utils/` holds shared stencil and convergence helpers. | — |
| `benchmarks/` | Performance benchmarks (roofline, throughput). | — |
| `replication/` | Formula register and replication targets. | `replication/formulas.md` |
| `derivations/` | SymPy derivation documents for physics capabilities (Lane B/C). | — |
| `docs/research/` | Research survey — code landscape, capabilities, licensing, V&V methodology. | `docs/research/index.md` |
| `pr-review/` | Adversarial PR review checklist and architecture stress-review checklist. | `pr-review/README.md` |
| `scripts/` | Agent health check, PR review wrappers, session startup, environment setup and activation. | `scripts/agent_health_check.sh` |
| `environment/` | Conda environment spec files and miniforge install target. | `environment/cosmic_foundry.yml` |

`theory/` defines an ABC tree; `computation/` and `mesh/` implement it at distance 1.
The full hierarchy is in [`ARCHITECTURE.md §Mathematical hierarchy`](ARCHITECTURE.md).

## Background

- [`docs/research/`](docs/research/index.md) — survey of the computational astrophysics
  code landscape that informs the design.
- [`ROADMAP.md`](ROADMAP.md) — epoch-by-epoch development plan.
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — architectural decisions.
- [`STATUS.md`](STATUS.md) — immediate implementation queue.

## License

[BSD 3-Clause](LICENSE)
