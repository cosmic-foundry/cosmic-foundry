"""Algorithm structure contracts for linear-solver selection."""

from __future__ import annotations

from cosmic_foundry.computation.algorithm_capabilities import (
    AffineComparisonPredicate,
    AlgorithmCapability,
    AlgorithmRegistry,
    AlgorithmRequest,
    AlgorithmStructureContract,
    ComparisonPredicate,
    CoveragePatch,
    EvidencePredicate,
    MembershipPredicate,
    ParameterDescriptor,
    StructuredPredicate,
    linear_solver_parameter_schema,
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


_LINEARITY_TOLERANCE = 1.0e-12
_CONDITION_LIMIT = 1.0e8


def _linear_system_predicates() -> tuple[StructuredPredicate, ...]:
    return (
        ComparisonPredicate("map_linearity_defect", "<=", _LINEARITY_TOLERANCE),
        AffineComparisonPredicate({"dim_x": 1.0, "dim_y": -1.0}, "==", 0.0),
        MembershipPredicate("residual_target_available", frozenset({True})),
        MembershipPredicate(
            "acceptance_relation",
            frozenset({"residual_below_tolerance"}),
        ),
    )


def _dense_matrix_predicates() -> tuple[MembershipPredicate, ...]:
    return (
        MembershipPredicate("matrix_representation_available", frozenset({True})),
        MembershipPredicate("linear_operator_matrix_available", frozenset({True})),
    )


def _budget_predicates() -> tuple[AffineComparisonPredicate, ...]:
    return (
        AffineComparisonPredicate(
            {"work_budget_fmas": 1.0, "assembly_cost_fmas": -1.0},
            ">=",
            0.0,
        ),
        AffineComparisonPredicate(
            {"work_budget_fmas": 1.0, "matvec_cost_fmas": -1.0},
            ">=",
            0.0,
        ),
        AffineComparisonPredicate(
            {
                "memory_budget_bytes": 1.0,
                "linear_operator_memory_bytes": -1.0,
            },
            ">=",
            0.0,
        ),
    )


def _owned_patch(
    name: str,
    owner: str,
    predicates: tuple[StructuredPredicate, ...],
    *,
    priority: int,
) -> CoveragePatch:
    return CoveragePatch(name, owner, "owned", predicates, priority=priority)


_LINEAR_SOLVER_OWNED_PATCHES: tuple[CoveragePatch, ...] = (
    _owned_patch(
        "dense_lu_square_full_rank_dense",
        "DenseLUSolver",
        _linear_system_predicates()
        + _dense_matrix_predicates()
        + _budget_predicates()
        + (
            ComparisonPredicate("singular_value_lower_bound", ">", 0.0),
            ComparisonPredicate("condition_estimate", "<=", _CONDITION_LIMIT),
            ComparisonPredicate("rhs_consistency_defect", "<=", _LINEARITY_TOLERANCE),
        ),
        priority=30,
    ),
    _owned_patch(
        "dense_svd_rank_deficient_dense",
        "DenseSVDSolver",
        _linear_system_predicates()
        + _dense_matrix_predicates()
        + _budget_predicates()
        + (
            ComparisonPredicate("nullity_estimate", ">", 0),
            ComparisonPredicate("rhs_consistency_defect", "<=", _LINEARITY_TOLERANCE),
        ),
        priority=20,
    ),
    _owned_patch(
        "dense_jacobi_strictly_diagonally_dominant",
        "DenseJacobiSolver",
        _linear_system_predicates()
        + _dense_matrix_predicates()
        + _budget_predicates()
        + (
            ComparisonPredicate("diagonal_nonzero_margin", ">", 0.0),
            ComparisonPredicate("diagonal_dominance_margin", ">", 0.0),
            ComparisonPredicate("condition_estimate", "<=", _CONDITION_LIMIT),
            ComparisonPredicate("rhs_consistency_defect", "<=", _LINEARITY_TOLERANCE),
        ),
        priority=25,
    ),
    _owned_patch(
        "dense_cg_well_conditioned_spd",
        "DenseCGSolver",
        _linear_system_predicates()
        + _budget_predicates()
        + (
            MembershipPredicate("operator_application_available", frozenset({True})),
            ComparisonPredicate("symmetry_defect", "<=", _LINEARITY_TOLERANCE),
            ComparisonPredicate("coercivity_lower_bound", ">", 0.0),
            ComparisonPredicate("singular_value_lower_bound", ">", 0.0),
            ComparisonPredicate("condition_estimate", "<=", _CONDITION_LIMIT),
            ComparisonPredicate("rhs_consistency_defect", "<=", _LINEARITY_TOLERANCE),
        ),
        priority=10,
    ),
    _owned_patch(
        "dense_gmres_matrix_free_nonsymmetric",
        "DenseGMRESSolver",
        _linear_system_predicates()
        + _budget_predicates()
        + (
            MembershipPredicate("matrix_representation_available", frozenset({False})),
            MembershipPredicate(
                "linear_operator_matrix_available",
                frozenset({False}),
            ),
            MembershipPredicate("operator_application_available", frozenset({True})),
            ComparisonPredicate("symmetry_defect", ">", _LINEARITY_TOLERANCE),
            ComparisonPredicate("singular_value_lower_bound", ">", 0.0),
            ComparisonPredicate("condition_estimate", "<=", _CONDITION_LIMIT),
            ComparisonPredicate("rhs_consistency_defect", "<=", _LINEARITY_TOLERANCE),
        ),
        priority=15,
    ),
)

_LINEAR_SOLVER_REJECTION_PATCHES: tuple[CoveragePatch, ...] = (
    CoveragePatch(
        "linear_solver_work_budget_below_operator_cost",
        "linear_solver_selector",
        "rejected",
        (
            ComparisonPredicate("map_linearity_defect", "<=", _LINEARITY_TOLERANCE),
            AffineComparisonPredicate(
                {"work_budget_fmas": 1.0, "matvec_cost_fmas": -1.0},
                "<",
                0.0,
            ),
        ),
    ),
    CoveragePatch(
        "linear_solver_memory_budget_below_operator_storage",
        "linear_solver_selector",
        "rejected",
        (
            ComparisonPredicate("map_linearity_defect", "<=", _LINEARITY_TOLERANCE),
            AffineComparisonPredicate(
                {
                    "memory_budget_bytes": 1.0,
                    "linear_operator_memory_bytes": -1.0,
                },
                "<",
                0.0,
            ),
        ),
    ),
    CoveragePatch(
        "linear_solver_unknown_condition",
        "linear_solver_selector",
        "rejected",
        (
            ComparisonPredicate("map_linearity_defect", "<=", _LINEARITY_TOLERANCE),
            EvidencePredicate("condition_estimate", frozenset({"unavailable"})),
        ),
    ),
)


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
        coverage_patches=(_LINEAR_SOLVER_OWNED_PATCHES[0],),
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
        coverage_patches=(_LINEAR_SOLVER_OWNED_PATCHES[1],),
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
        coverage_patches=(_LINEAR_SOLVER_OWNED_PATCHES[2],),
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
        coverage_patches=(_LINEAR_SOLVER_OWNED_PATCHES[3],),
    ),
    LinearSolverCapability(
        "dense_gmres_iteration",
        "DenseGMRESSolver",
        "iterative_solver",
        _contract(
            requires=("linear_operator", "nonsingular"),
            provides=("solve", "iterative", "krylov", "matrix_free", "general"),
        ),
        coverage_patches=(_LINEAR_SOLVER_OWNED_PATCHES[4],),
    ),
)


LINEAR_SOLVER_REGISTRY = LinearSolverRegistry(_CAPABILITIES)


def linear_solver_capabilities() -> tuple[LinearSolverCapability, ...]:
    """Return declared linear-solver algorithm capabilities."""
    return LINEAR_SOLVER_REGISTRY.capabilities


def select_linear_solver(request: LinearSolverRequest) -> LinearSolverCapability:
    """Select a linear-solver implementation declaration by capability."""
    return LINEAR_SOLVER_REGISTRY.select(request)


def linear_solver_coverage_patches() -> tuple[CoveragePatch, ...]:
    """Return descriptor-space coverage patches for linear-solver selection."""
    patches: list[CoveragePatch] = []
    for capability in LINEAR_SOLVER_REGISTRY.capabilities:
        patches.extend(capability.coverage_patches)
    patches.extend(_LINEAR_SOLVER_REJECTION_PATCHES)
    return tuple(patches)


def select_linear_solver_for_descriptor(
    descriptor: ParameterDescriptor,
) -> LinearSolverCapability:
    """Select a linear solver by parameter-space descriptor coverage."""
    schema = linear_solver_parameter_schema()
    patches = linear_solver_coverage_patches()
    status = schema.cell_status(descriptor, patches)
    if status == "invalid":
        raise ValueError(f"invalid linear-solver descriptor {descriptor!r}")
    if status == "rejected":
        raise ValueError(f"rejected linear-solver descriptor {descriptor!r}")
    if status == "uncovered":
        raise ValueError(f"no linear solver covers descriptor {descriptor!r}")

    owners = {
        capability.implementation: capability
        for capability in LINEAR_SOLVER_REGISTRY.capabilities
    }
    matches = tuple(
        patch
        for patch in patches
        if patch.status == "owned" and patch.contains(descriptor)
    )
    ranked = sorted(
        matches,
        key=lambda patch: patch.priority if patch.priority is not None else 1_000_000,
    )
    if len(ranked) > 1 and ranked[0].priority == ranked[1].priority:
        names = ", ".join(patch.name for patch in ranked)
        raise ValueError(f"ambiguous linear-solver descriptor priority: {names}")
    return owners[ranked[0].owner]


__all__ = [
    "LinearSolverCapability",
    "linear_solver_capabilities",
    "linear_solver_coverage_patches",
    "LinearSolverRegistry",
    "LinearSolverRequest",
    "LINEAR_SOLVER_REGISTRY",
    "select_linear_solver",
    "select_linear_solver_for_descriptor",
]
