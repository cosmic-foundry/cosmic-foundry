"""Matrix decomposition classes: ABCs and concrete algorithms."""

from cosmic_foundry.computation.decompositions.capabilities import (
    DECOMPOSITION_COVERAGE_REGIONS,
    decomposition_coverage_regions,
    select_decomposition_for_descriptor,
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
    "decomposition_coverage_regions",
    "DECOMPOSITION_COVERAGE_REGIONS",
    "Decomposition",
    "DecomposedTensor",
    "Factorization",
    "LUDecomposedTensor",
    "LUFactorization",
    "select_decomposition_for_descriptor",
    "SVDDecomposedTensor",
    "SVDFactorization",
]
