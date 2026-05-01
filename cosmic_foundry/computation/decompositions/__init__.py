"""Matrix decomposition classes: ABCs and concrete algorithms."""

from cosmic_foundry.computation.decompositions.capabilities import (
    DecompositionCapability,
    DecompositionRegistry,
    DecompositionRequest,
    decomposition_capabilities,
    select_decomposition,
)
from cosmic_foundry.computation.decompositions.decomposition import (
    DecomposedTensor,
    Decomposition,
)
from cosmic_foundry.computation.decompositions.factorization import Factorization
from cosmic_foundry.computation.decompositions.lu_factorization import (
    LUDecomposedTensor,
    LUFactorization,
)
from cosmic_foundry.computation.decompositions.svd_factorization import (
    SVDDecomposedTensor,
    SVDFactorization,
)

__all__ = [
    "decomposition_capabilities",
    "Decomposition",
    "DecompositionCapability",
    "DecomposedTensor",
    "DecompositionRegistry",
    "DecompositionRequest",
    "Factorization",
    "LUDecomposedTensor",
    "LUFactorization",
    "select_decomposition",
    "SVDDecomposedTensor",
    "SVDFactorization",
]
