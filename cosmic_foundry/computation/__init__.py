"""Computation layer."""

import jax

# Guarantee float64 precision before any JAX computation is JIT-compiled.
jax.config.update("jax_enable_x64", True)

from cosmic_foundry.computation.tensor import Real, Tensor, einsum  # noqa: E402

__all__ = [
    "DenseJacobiSolver",
    "DenseLUSolver",
    "LinearSolver",
    "Real",
    "Tensor",
    "einsum",
]
