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
from cosmic_foundry.computation.tensor import (  # noqa: E402
    MaterializationError,
    Real,
    Tensor,
    arange,
    einsum,
    where,
)

# Register Tensor as a JAX pytree so jax.lax.fori_loop, jax.lax.while_loop,
# and other JAX transformations can flatten state pytrees containing Tensors
# into raw JAX arrays and re-wrap them on the way out.  The leaf is the raw
# backend value;
# the backend instance is carried as static aux data.
jax.tree_util.register_pytree_node(
    Tensor,
    lambda t: ((t._value,), t._backend),
    lambda backend, leaves: Tensor._wrap(leaves[0], backend),
)

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
    "MaterializationError",
    "NumpyBackend",
    "PythonBackend",
    "Real",
    "Tensor",
    "arange",
    "einsum",
    "get_default_backend",
    "set_default_backend",
    "where",
]
