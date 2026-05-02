"""Autodiscovered linear-solver coverage aggregation."""

from __future__ import annotations

from importlib import import_module
from pkgutil import iter_modules
from types import ModuleType

from cosmic_foundry.computation.algorithm_capabilities import (
    AffineComparisonPredicate,
    ComparisonPredicate,
    CoveragePatch,
    EvidencePredicate,
    ParameterDescriptor,
    linear_solver_parameter_schema,
)
from cosmic_foundry.computation.solvers.coverage import (
    LINEARITY_TOLERANCE,
    LinearSolverCoverage,
    coverage,
)


def _solver_package_modules() -> tuple[ModuleType, ...]:
    package = import_module(__package__ or "cosmic_foundry.computation.solvers")
    package_path = getattr(package, "__path__", ())
    modules: list[ModuleType] = []
    for module_info in sorted(iter_modules(package_path), key=lambda info: info.name):
        if module_info.name.startswith("_") or module_info.name == "capabilities":
            continue
        modules.append(import_module(f"{package.__name__}.{module_info.name}"))
    return tuple(modules)


def _discovered_coverages() -> tuple[LinearSolverCoverage, ...]:
    coverages: list[LinearSolverCoverage] = []
    for module in _solver_package_modules():
        for item in module.__dict__.values():
            if not isinstance(item, type) or item.__module__ != module.__name__:
                continue
            predicates = getattr(item, "linear_solver_coverage", None)
            if predicates is not None:
                coverages.append(
                    coverage(
                        item,
                        coverage_predicates=predicates,
                    )
                )
    return tuple(coverages)


LINEAR_SOLVER_COVERAGES = _discovered_coverages()


def linear_solver_coverages() -> tuple[LinearSolverCoverage, ...]:
    """Return autodiscovered linear-solver descriptor coverage records."""
    return LINEAR_SOLVER_COVERAGES


def linear_solver_coverage_patches() -> tuple[CoveragePatch, ...]:
    """Return autodiscovered descriptor-space coverage patches."""
    patches: list[CoveragePatch] = []
    for record in LINEAR_SOLVER_COVERAGES:
        patches.extend(record.coverage_patches)
    patches.extend(selector_rejection_patches())
    return tuple(patches)


def selector_rejection_patches() -> tuple[CoveragePatch, ...]:
    """Return selector-level rejection regions not owned by implementations."""
    return (
        CoveragePatch(
            "linear_solver_work_budget_below_operator_cost",
            "linear_solver_selector",
            "rejected",
            (
                ComparisonPredicate(
                    "map_linearity_defect",
                    "<=",
                    LINEARITY_TOLERANCE,
                ),
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
                ComparisonPredicate(
                    "map_linearity_defect",
                    "<=",
                    LINEARITY_TOLERANCE,
                ),
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
                ComparisonPredicate(
                    "map_linearity_defect",
                    "<=",
                    LINEARITY_TOLERANCE,
                ),
                EvidencePredicate("condition_estimate", frozenset({"unavailable"})),
            ),
        ),
    )


def select_linear_solver_for_descriptor(
    descriptor: ParameterDescriptor,
) -> LinearSolverCoverage:
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

    owners = {record.implementation: record for record in LINEAR_SOLVER_COVERAGES}
    matches = tuple(
        patch
        for patch in patches
        if patch.status == "owned" and patch.contains(descriptor)
    )
    if len(matches) > 1:
        names = ", ".join(patch.name for patch in matches)
        raise ValueError(f"ambiguous linear-solver descriptor coverage: {names}")
    return owners[matches[0].owner]


__all__ = [
    "LinearSolverCoverage",
    "linear_solver_coverages",
    "linear_solver_coverage_patches",
    "LINEAR_SOLVER_COVERAGES",
    "select_linear_solver_for_descriptor",
]
