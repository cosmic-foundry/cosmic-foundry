"""Cosmic Foundry — high-performance computational astrophysics engine."""

import jax

# The package guarantees float64 precision at import time.
# Must be set before any JAX computation is JIT-compiled.
jax.config.update("jax_enable_x64", True)

from cosmic_foundry._version import __version__  # noqa: E402
from cosmic_foundry.theory import (  # noqa: E402
    CovectorField,
    DifferentialForm,
    DifferentialOperator,
    Field,
    Function,
    IndexedSet,
    MetricTensor,
    PseudoRiemannianManifold,
    Region,
    RiemannianManifold,
    ScalarField,
    Set,
    SmoothManifold,
    SymmetricTensorField,
    TensorField,
    VectorField,
)

__all__ = [
    "__version__",
    # fields
    "CovectorField",
    "DifferentialForm",
    "DifferentialOperator",
    "Field",
    "MetricTensor",
    "Region",
    "ScalarField",
    "SymmetricTensorField",
    "TensorField",
    "VectorField",
    # theory
    "Function",
    "IndexedSet",
    "PseudoRiemannianManifold",
    "RiemannianManifold",
    "Set",
    "SmoothManifold",
]
