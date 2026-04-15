"""Cosmic Foundry — high-performance computational astrophysics engine."""

import jax

# ADR-0009: the package guarantees float64 precision at import time.
# Must be set before any JAX computation is JIT-compiled.
jax.config.update("jax_enable_x64", True)

from cosmic_foundry._version import __version__  # noqa: E402

__all__ = ["__version__"]
