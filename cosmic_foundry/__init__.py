"""Cosmic Foundry — high-performance computational astrophysics engine."""

from cosmic_foundry._version import __version__  # noqa: E402
from cosmic_foundry.theory.continuous import (  # noqa: E402
    DifferentialForm,
    DifferentialOperator,
    Field,
    MetricTensor,
    PseudoRiemannianManifold,
    SymmetricTensorField,
    TensorField,
)
from cosmic_foundry.theory.foundation import (  # noqa: E402
    Function,
    IndexedSet,
    Set,
)

__all__ = [
    "__version__",
    # fields
    "DifferentialForm",
    "DifferentialOperator",
    "Field",
    "MetricTensor",
    "SymmetricTensorField",
    "TensorField",
    # theory
    "Function",
    "IndexedSet",
    "PseudoRiemannianManifold",
    "Set",
]
