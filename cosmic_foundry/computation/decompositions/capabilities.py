"""Algorithm structure contracts for decomposition selection."""

from __future__ import annotations

from cosmic_foundry.computation.algorithm_capabilities import (
    AlgorithmCapability,
    AlgorithmRegistry,
    AlgorithmRequest,
    AlgorithmStructureContract,
)

DecompositionCapability = AlgorithmCapability
DecompositionRegistry = AlgorithmRegistry
DecompositionRequest = AlgorithmRequest


def _contract(
    *,
    requires: tuple[str, ...],
    provides: tuple[str, ...],
) -> AlgorithmStructureContract:
    return AlgorithmStructureContract(frozenset(requires), frozenset(provides))


_CAPABILITIES: tuple[DecompositionCapability, ...] = (
    DecompositionCapability(
        "lu_factorization",
        "LUFactorization",
        "factorization",
        _contract(
            requires=("dense_matrix", "square_matrix", "full_rank"),
            provides=("decompose", "factorize", "direct_solve", "pivoting", "exact"),
        ),
        priority=10,
    ),
    DecompositionCapability(
        "svd_factorization",
        "SVDFactorization",
        "factorization",
        _contract(
            requires=("dense_matrix",),
            provides=(
                "decompose",
                "factorize",
                "direct_solve",
                "least_squares",
                "minimum_norm",
                "rank_deficient",
                "singular_values",
            ),
        ),
    ),
)


DECOMPOSITION_REGISTRY = DecompositionRegistry(_CAPABILITIES)


def decomposition_capabilities() -> tuple[DecompositionCapability, ...]:
    """Return declared decomposition algorithm capabilities."""
    return DECOMPOSITION_REGISTRY.capabilities


def select_decomposition(request: DecompositionRequest) -> DecompositionCapability:
    """Select a decomposition implementation declaration by capability."""
    return DECOMPOSITION_REGISTRY.select(request)


__all__ = [
    "DecompositionCapability",
    "decomposition_capabilities",
    "DECOMPOSITION_REGISTRY",
    "DecompositionRegistry",
    "DecompositionRequest",
    "select_decomposition",
]
