# Cosmic Foundry

A computational astrophysics engine for multi-physics simulation.

```{toctree}
:maxdepth: 2
:caption: Getting started

getting-started
contributing
coding-standards
```

```{toctree}
:maxdepth: 2
:caption: Theory

theory/index
```

```{toctree}
:maxdepth: 1
:caption: API reference

api/index
```

```{toctree}
:maxdepth: 1
:caption: Gallery

gallery/index
accessibility
```

```{toctree}
:maxdepth: 1
:caption: Architecture

adr/index
```

## Overview

Cosmic Foundry is a Python engine for numerical astrophysics, targeting
the capability set surveyed in
[`RESEARCH.md`](https://github.com/cosmic-foundry/cosmic-foundry/blob/main/RESEARCH.md)
— hydrodynamics through numerical relativity, radiation transport, gravity,
N-body, microphysics, and cosmology. It is being built incrementally through
a roadmap of staged capabilities toward a code that can replicate published
astrophysics results.

For the full design survey and development plan, see
[`RESEARCH.md`](https://github.com/cosmic-foundry/cosmic-foundry/blob/main/RESEARCH.md)
and the
[development roadmap](https://github.com/cosmic-foundry/cosmic-foundry/blob/main/ROADMAP.md).
Architectural decisions are recorded in
[`adr/README.md`](https://github.com/cosmic-foundry/cosmic-foundry/blob/main/adr/README.md).
The numerical kernels run on JAX; see
[ADR-0002](https://github.com/cosmic-foundry/cosmic-foundry/blob/main/adr/object-level/ADR-0002-jax-primary-kernel-backend.md).
