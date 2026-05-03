"""Autodiscovered linear-solver coverage aggregation."""

from __future__ import annotations

from importlib import import_module
from pkgutil import iter_modules
from types import ModuleType

from cosmic_foundry.computation.algorithm_capabilities import (
    CoverageRegion,
    ParameterDescriptor,
    StructuredPredicate,
    linear_solver_parameter_schema,
)
from cosmic_foundry.computation.solvers.coverage import (
    budget_predicates,
    coverage,
    dense_matrix_predicates,
    linear_system_predicates,
    matrix_free_operator_predicates,
)
from cosmic_foundry.computation.solvers.direct_solver import DirectSolver
from cosmic_foundry.computation.solvers.iterative_solver import KrylovSolver
from cosmic_foundry.computation.solvers.linear_solver import LinearSolver


def _solver_package_modules() -> tuple[ModuleType, ...]:
    package = import_module(__package__ or "cosmic_foundry.computation.solvers")
    package_path = getattr(package, "__path__", ())
    modules: list[ModuleType] = []
    for module_info in sorted(iter_modules(package_path), key=lambda info: info.name):
        if module_info.name.startswith("_") or module_info.name == "capabilities":
            continue
        modules.append(import_module(f"{package.__name__}.{module_info.name}"))
    return tuple(modules)


def _inherited_coverage_predicates(owner: type) -> tuple[StructuredPredicate, ...]:
    predicates: tuple[StructuredPredicate, ...] = ()
    if issubclass(owner, LinearSolver):
        predicates += linear_system_predicates() + budget_predicates()
    if issubclass(owner, DirectSolver):
        decomposition_type = getattr(owner, "decomposition_type", None)
        predicates += dense_matrix_predicates()
        if decomposition_type is not None:
            predicates += decomposition_type.linear_solve_certificate
    if issubclass(owner, KrylovSolver):
        predicates += matrix_free_operator_predicates()
    return predicates


def _owns_coverage(owner: type) -> bool:
    return (
        "linear_solver_coverage" in owner.__dict__
        or getattr(owner, "decomposition_type", None) is not None
    )


def _discovered_coverage_regions() -> tuple[CoverageRegion, ...]:
    regions: list[CoverageRegion] = []
    for module in _solver_package_modules():
        for item in module.__dict__.values():
            if not isinstance(item, type) or item.__module__ != module.__name__:
                continue
            if _owns_coverage(item):
                regions.append(
                    coverage(
                        item,
                        coverage_predicates=_inherited_coverage_predicates(item)
                        + getattr(item, "linear_solver_coverage", ()),
                    )
                )
    return tuple(regions)


LINEAR_SOLVER_COVERAGE_REGIONS = _discovered_coverage_regions()


def linear_solver_coverage_regions() -> tuple[CoverageRegion, ...]:
    """Return autodiscovered descriptor-space coverage regions."""
    return LINEAR_SOLVER_COVERAGE_REGIONS


def select_linear_solver_for_descriptor(
    descriptor: ParameterDescriptor,
) -> type:
    """Select a linear solver by parameter-space descriptor coverage."""
    schema = linear_solver_parameter_schema()
    regions = linear_solver_coverage_regions()
    region = schema.covering_region(descriptor, regions)
    if region is None:
        raise ValueError(f"no linear solver covers descriptor {descriptor!r}")

    return region.owner


__all__ = [
    "linear_solver_coverage_regions",
    "LINEAR_SOLVER_COVERAGE_REGIONS",
    "select_linear_solver_for_descriptor",
]
