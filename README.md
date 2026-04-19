# Cosmic Foundry

A computational astrophysics engine for multi-physics simulation.

Cosmic Foundry is a Python engine for numerical astrophysics, targeting
the capability set surveyed in [`docs/research/`](docs/research/index.md) — hydrodynamics
through numerical relativity, radiation transport, gravity, N-body,
microphysics, and cosmology. It is being built incrementally toward a
code that can replicate published astrophysics results. JAX powers the
numerical kernels (see `ARCHITECTURE.md` for the technology baseline).

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

## Background

- [`docs/research/`](docs/research/index.md) — survey of the computational astrophysics
  code landscape that informs the design.
- [`ROADMAP.md`](ROADMAP.md) — epoch-by-epoch development plan.
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — architectural decisions.

## License

[BSD 3-Clause](LICENSE)
