# Cosmic Foundry

Cosmic Foundry is a general-purpose PDE simulation engine, optimized
for astrophysical use cases. The foundational commitments governing its
design are in [**Architectural basis**](ARCHITECTURE.md#architectural-basis).

## Quick start

```bash
# Clone the repo
git clone git@github.com:cosmic-foundry/cosmic-foundry.git
cd cosmic-foundry

# One-time setup (~5 min)
bash scripts/setup_environment.sh

# Start an agent session (activates environment automatically)
./scripts/start_agent.sh claude   # or gemini / codex

# Or activate manually and work directly
source scripts/activate_environment.sh
```

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
- [`STATUS.md`](STATUS.md) — immediate implementation queue.

## License

[BSD 3-Clause](LICENSE)
