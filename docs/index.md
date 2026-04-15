# Cosmic Foundry

A high-performance computational astrophysics engine built on JAX.

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

Cosmic Foundry is a Python engine for astrophysics simulations. It uses JAX
for JIT-compiled, differentiable kernels that run on CPU or GPU without code
changes.

For background on the design, see
[`RESEARCH.md`](https://github.com/cosmic-foundry/cosmic-foundry/blob/main/RESEARCH.md)
and the
[development roadmap](https://github.com/cosmic-foundry/cosmic-foundry/blob/main/ROADMAP.md).
Architectural decisions are recorded in
[`adr/README.md`](https://github.com/cosmic-foundry/cosmic-foundry/blob/main/adr/README.md).
