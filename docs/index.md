# Cosmic Foundry

A computational astrophysics engine for multi-physics simulation.

```{toctree}
:maxdepth: 1
:caption: API reference

api/index
```

```{toctree}
:maxdepth: 1
:caption: Research survey

research/index
```

## Overview

Cosmic Foundry is a Python engine for numerical astrophysics. The simulation
capability sequence — hydrodynamics through numerical relativity, radiation
transport, gravity, N-body, microphysics, and cosmology — is defined in
[`ROADMAP.md`](https://github.com/cosmic-foundry/cosmic-foundry/blob/main/ROADMAP.md).
Current development status and the immediate next steps for both the simulation
and verification tracks are in
[`STATUS.md`](https://github.com/cosmic-foundry/cosmic-foundry/blob/main/STATUS.md).

## Architecture

All live architectural decisions are recorded as one-paragraph claims in
[`ARCHITECTURE.md`](https://github.com/cosmic-foundry/cosmic-foundry/blob/main/ARCHITECTURE.md).
That file is the single authoritative source for platform structure, the
kernel and field abstractions, the operator model, the mesh layer, I/O,
diagnostics, and the documentation and visualization stacks.

## Development

The full developer workflow — environment setup, branch and PR discipline,
commit size, physics capability lanes, verification standards, and epoch
retrospectives — is in
[`DEVELOPMENT.md`](https://github.com/cosmic-foundry/cosmic-foundry/blob/main/DEVELOPMENT.md).
Agent-specific guidelines (session startup, PR review, lane selection) are in
[`AI.md`](https://github.com/cosmic-foundry/cosmic-foundry/blob/main/AI.md).
