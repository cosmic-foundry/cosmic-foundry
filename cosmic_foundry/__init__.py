"""Cosmic Foundry — high-performance computational astrophysics engine."""

import jax

# The package guarantees float64 precision at import time.
# Must be set before any JAX computation is JIT-compiled.
jax.config.update("jax_enable_x64", True)

from cosmic_foundry._version import __version__  # noqa: E402
from cosmic_foundry.computation.array import Array  # noqa: E402
from cosmic_foundry.computation.extent import Extent  # noqa: E402
from cosmic_foundry.theory import (  # noqa: E402
    Discretization,
    Field,
    Function,
    IndexedSet,
    LocatedDiscretization,
    ModalDiscretization,
    PseudoRiemannianManifold,
    RiemannianManifold,
    ScalarField,
    Set,
    Sink,
    SmoothManifold,
    Source,
    TensorField,
)

__all__ = [
    "__version__",
    # computation
    "Array",
    "Extent",
    # fields
    "Field",
    "ScalarField",
    "TensorField",
    # theory
    "Discretization",
    "Function",
    "IndexedSet",
    "LocatedDiscretization",
    "ModalDiscretization",
    "PseudoRiemannianManifold",
    "RiemannianManifold",
    "Set",
    "Sink",
    "SmoothManifold",
    "Source",
]
