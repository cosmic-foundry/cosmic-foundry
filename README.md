# Cosmic Foundry

A high-performance computational astrophysics engine built on JAX.

Cosmic Foundry is a Python engine for astrophysics simulations. It uses
JAX for JIT-compiled, differentiable kernels that run on CPU or GPU without
code changes. The engine is being built incrementally through a series of
epochs, starting from this bootstrap (Epoch 0) and growing toward a full
multi-physics code that replicates published astrophysics results.

## Quick start

```bash
# Clone the fork
git clone git@github.com:<your-fork>/cosmic-foundry.git
cd cosmic-foundry

# One-time setup (~5 min)
bash environment/setup_environment.sh

# Start an agent session (activates environment automatically)
./scripts/start_agent.sh claude   # or gemini / codex

# Or activate manually and work directly
source environment/activate_environment.sh
pip install -e .[dev]
cosmic-foundry hello
```

`cosmic-foundry hello` prints the JAX backend, device list, and a JIT
smoke-test result. If it exits 0, the environment is correctly wired.

## Documentation

Build the docs locally after `pip install -e .[dev,docs]`:

```bash
sphinx-build -W docs docs/_build/html
```

## Background

- [`RESEARCH.md`](RESEARCH.md) — survey of the computational astrophysics
  code landscape that informs the design.
- [`ROADMAP.md`](ROADMAP.md) — epoch-by-epoch development plan.
- [`adr/README.md`](adr/README.md) — architectural decision records.

## License

[BSD 3-Clause](LICENSE)
