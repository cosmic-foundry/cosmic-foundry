"""Computation layer."""

import jax

# Guarantee float64 precision before any JAX computation is JIT-compiled.
jax.config.update("jax_enable_x64", True)

from cosmic_foundry.computation.backends import (  # noqa: E402
    Backend,
    JaxBackend,
    NumpyBackend,
    PythonBackend,
    get_default_backend,
    set_default_backend,
)
from cosmic_foundry.computation.tensor import Real, Tensor, einsum  # noqa: E402

__all__ = [
    "Backend",
    "DenseJacobiSolver",
    "DenseLUSolver",
    "DirectSolver",
    "Factorization",
    "FactoredMatrix",
    "IterativeSolver",
    "JaxBackend",
    "LUFactorization",
    "LUFactoredMatrix",
    "LinearSolver",
    "NumpyBackend",
    "PythonBackend",
    "Real",
    "Tensor",
    "einsum",
    "get_default_backend",
    "set_default_backend",
]
