"""Algorithm structure contracts for linear-solver selection."""

from __future__ import annotations

from cosmic_foundry.computation.algorithm_capabilities import (
    AlgorithmCapability,
    AlgorithmRegistry,
    AlgorithmRequest,
    AlgorithmStructureContract,
)

LinearSolverCapability = AlgorithmCapability
LinearSolverRegistry = AlgorithmRegistry
LinearSolverRequest = AlgorithmRequest


def _contract(
    *,
    requires: tuple[str, ...],
    provides: tuple[str, ...],
) -> AlgorithmStructureContract:
    return AlgorithmStructureContract(frozenset(requires), frozenset(provides))


_CAPABILITIES: tuple[LinearSolverCapability, ...] = (
    LinearSolverCapability(
        "dense_lu_direct",
        "DenseLUSolver",
        "direct_solver",
        _contract(
            requires=("dense_operator", "square_system", "full_rank"),
            provides=("solve", "direct", "exact", "factorized_dense"),
        ),
        priority=10,
    ),
    LinearSolverCapability(
        "dense_svd_direct",
        "DenseSVDSolver",
        "direct_solver",
        _contract(
            requires=("dense_operator",),
            provides=(
                "solve",
                "direct",
                "least_squares",
                "minimum_norm",
                "rank_deficient",
                "factorized_dense",
            ),
        ),
    ),
    LinearSolverCapability(
        "generic_direct",
        "DirectSolver",
        "direct_solver",
        _contract(
            requires=("linear_operator", "decomposition"),
            provides=("solve", "direct", "assembled_matrix"),
        ),
    ),
    LinearSolverCapability(
        "dense_jacobi_iteration",
        "DenseJacobiSolver",
        "iterative_solver",
        _contract(
            requires=("linear_operator", "diagonal", "row_abs_sums"),
            provides=("solve", "iterative", "stationary", "matrix_free"),
        ),
    ),
    LinearSolverCapability(
        "dense_gauss_seidel_iteration",
        "DenseGaussSeidelSolver",
        "iterative_solver",
        _contract(
            requires=("linear_operator", "square_system"),
            provides=("solve", "iterative", "stationary", "assembled_matrix"),
        ),
    ),
    LinearSolverCapability(
        "dense_cg_iteration",
        "DenseCGSolver",
        "iterative_solver",
        _contract(
            requires=("linear_operator", "symmetric_positive_definite"),
            provides=("solve", "iterative", "krylov", "matrix_free", "spd"),
        ),
        priority=10,
    ),
    LinearSolverCapability(
        "dense_gmres_iteration",
        "DenseGMRESSolver",
        "iterative_solver",
        _contract(
            requires=("linear_operator", "nonsingular"),
            provides=("solve", "iterative", "krylov", "matrix_free", "general"),
        ),
    ),
)


LINEAR_SOLVER_REGISTRY = LinearSolverRegistry(_CAPABILITIES)


def linear_solver_capabilities() -> tuple[LinearSolverCapability, ...]:
    """Return declared linear-solver algorithm capabilities."""
    return LINEAR_SOLVER_REGISTRY.capabilities


def select_linear_solver(request: LinearSolverRequest) -> LinearSolverCapability:
    """Select a linear-solver implementation declaration by capability."""
    return LINEAR_SOLVER_REGISTRY.select(request)


__all__ = [
    "LinearSolverCapability",
    "linear_solver_capabilities",
    "LinearSolverRegistry",
    "LinearSolverRequest",
    "LINEAR_SOLVER_REGISTRY",
    "select_linear_solver",
]
