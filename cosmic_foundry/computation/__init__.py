"""Computation layer."""

import jax

# Guarantee float64 precision before any JAX computation is JIT-compiled.
jax.config.update("jax_enable_x64", True)

from cosmic_foundry.computation.abstract_backend import AbstractBackend  # noqa: E402
from cosmic_foundry.computation.abstract_value import (  # noqa: E402
    AbstractValue,
    JitIncompatibleError,
)
from cosmic_foundry.computation.backends import (  # noqa: E402
    Backend,
    JaxBackend,
    NumpyBackend,
    PythonBackend,
    get_default_backend,
    set_default_backend,
)
from cosmic_foundry.computation.tensor import (  # noqa: E402
    Real,
    Tensor,
    arange,
    einsum,
    where,
)
from cosmic_foundry.computation.tracing_backend import TracingBackend  # noqa: E402

__all__ = [
    "AbstractBackend",
    "AbstractValue",
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
    "arange",
    "einsum",
    "JitIncompatibleError",
    "TracingBackend",
    "get_default_backend",
    "set_default_backend",
    "where",
]
