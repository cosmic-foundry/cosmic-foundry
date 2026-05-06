"""Autodiscovered linear-solver coverage aggregation."""

from __future__ import annotations

from importlib import import_module
from pkgutil import iter_modules
from types import ModuleType

from cosmic_foundry.computation.algorithm_capabilities import (
    ComparisonPredicate,
    CoverageRegion,
    LinearSolverField,
    ParameterDescriptor,
    StructuredPredicate,
    linear_solver_parameter_schema,
    solve_relation_parameter_schema,
)
from cosmic_foundry.computation.decompositions.decomposition import Decomposition
from cosmic_foundry.computation.solvers.coverage import (
    LINEARITY_TOLERANCE,
    budget_predicates,
    coverage,
    dense_matrix_predicates,
    eigenpair_predicates,
    least_squares_predicates,
    linear_system_predicates,
    matrix_free_operator_predicates,
)
from cosmic_foundry.computation.solvers.direct_solver import DirectSolver
from cosmic_foundry.computation.solvers.iterative_solver import KrylovSolver
from cosmic_foundry.computation.solvers.least_squares_solver import LeastSquaresSolver
from cosmic_foundry.computation.solvers.linear_solver import LinearSolver
from cosmic_foundry.computation.solvers.spectral_solver import SpectralSolver


def _linear_solve_regions_from_decomposition(
    decomposition_type: type[Decomposition],
) -> tuple[tuple[StructuredPredicate, ...], ...]:
    return decomposition_type.factorization_feasibility_regions


def _solver_package_modules() -> tuple[ModuleType, ...]:
    package = import_module(__package__ or "cosmic_foundry.computation.solvers")
    package_path = getattr(package, "__path__", ())
    modules: list[ModuleType] = []
    for module_info in sorted(iter_modules(package_path), key=lambda info: info.name):
        if module_info.name.startswith("_") or module_info.name == "capabilities":
            continue
        modules.append(import_module(f"{package.__name__}.{module_info.name}"))
    return tuple(modules)


def _inherited_coverage_regions(
    owner: type,
) -> tuple[tuple[StructuredPredicate, ...], ...]:
    predicates: tuple[StructuredPredicate, ...] = ()
    if issubclass(owner, LinearSolver):
        predicates += linear_system_predicates() + budget_predicates()
    if issubclass(owner, DirectSolver):
        decomposition_type = getattr(owner, "decomposition_type", None)
        predicates += dense_matrix_predicates() + (
            ComparisonPredicate(
                LinearSolverField.RHS_CONSISTENCY_DEFECT,
                "<=",
                LINEARITY_TOLERANCE,
            ),
        )
        if decomposition_type is not None:
            return tuple(
                predicates + region
                for region in _linear_solve_regions_from_decomposition(
                    decomposition_type
                )
            )
    if issubclass(owner, KrylovSolver):
        predicates += matrix_free_operator_predicates()
    return (predicates,)


def _owns_coverage(owner: type) -> bool:
    return (
        "linear_solver_coverage" in owner.__dict__
        or getattr(owner, "decomposition_type", None) is not None
    )


def _owns_root_coverage(owner: type) -> bool:
    return "root_solver_coverage" in owner.__dict__


def _owns_least_squares_coverage(owner: type) -> bool:
    return issubclass(owner, LeastSquaresSolver) and owner is not LeastSquaresSolver


def _owns_spectral_coverage(owner: type) -> bool:
    return issubclass(owner, SpectralSolver) and owner is not SpectralSolver


def _discovered_coverage_regions() -> tuple[CoverageRegion, ...]:
    regions: list[CoverageRegion] = []
    for module in _solver_package_modules():
        for item in module.__dict__.values():
            if not isinstance(item, type) or item.__module__ != module.__name__:
                continue
            if _owns_coverage(item):
                regions.extend(
                    coverage(
                        item,
                        coverage_predicates=predicates
                        + getattr(item, "linear_solver_coverage", ()),
                    )
                    for predicates in _inherited_coverage_regions(item)
                )
    return tuple(regions)


def _discovered_least_squares_coverage_regions() -> tuple[CoverageRegion, ...]:
    regions: list[CoverageRegion] = []
    for module in _solver_package_modules():
        for item in module.__dict__.values():
            if not isinstance(item, type) or item.__module__ != module.__name__:
                continue
            if _owns_least_squares_coverage(item):
                regions.append(
                    coverage(item, coverage_predicates=least_squares_predicates())
                )
    return tuple(regions)


def _discovered_root_coverage_regions() -> tuple[CoverageRegion, ...]:
    regions: list[CoverageRegion] = []
    for module in _solver_package_modules():
        for item in module.__dict__.values():
            if not isinstance(item, type) or item.__module__ != module.__name__:
                continue
            if _owns_root_coverage(item):
                regions.extend(
                    coverage(item, coverage_predicates=predicates)
                    for predicates in getattr(item, "root_solver_coverage", ())
                )
    return tuple(regions)


def _discovered_spectral_coverage_regions() -> tuple[CoverageRegion, ...]:
    regions: list[CoverageRegion] = []
    for module in _solver_package_modules():
        for item in module.__dict__.values():
            if not isinstance(item, type) or item.__module__ != module.__name__:
                continue
            if _owns_spectral_coverage(item):
                regions.append(
                    coverage(item, coverage_predicates=eigenpair_predicates())
                )
    return tuple(regions)


LINEAR_SOLVER_COVERAGE_REGIONS = _discovered_coverage_regions()
LEAST_SQUARES_SOLVER_COVERAGE_REGIONS = _discovered_least_squares_coverage_regions()
ROOT_SOLVER_COVERAGE_REGIONS = _discovered_root_coverage_regions()
SPECTRAL_SOLVER_COVERAGE_REGIONS = _discovered_spectral_coverage_regions()


def linear_solver_coverage_regions() -> tuple[CoverageRegion, ...]:
    """Return autodiscovered descriptor-space coverage regions."""
    return LINEAR_SOLVER_COVERAGE_REGIONS


def root_solver_coverage_regions() -> tuple[CoverageRegion, ...]:
    """Return autodiscovered nonlinear-root coverage regions."""
    return ROOT_SOLVER_COVERAGE_REGIONS


def least_squares_solver_coverage_regions() -> tuple[CoverageRegion, ...]:
    """Return autodiscovered least-squares objective coverage regions."""
    return LEAST_SQUARES_SOLVER_COVERAGE_REGIONS


def spectral_solver_coverage_regions() -> tuple[CoverageRegion, ...]:
    """Return autodiscovered spectral solve-relation coverage regions."""
    return SPECTRAL_SOLVER_COVERAGE_REGIONS


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


def select_least_squares_solver_for_descriptor(
    descriptor: ParameterDescriptor,
) -> type:
    """Select a least-squares solver by primitive solve-relation coverage."""
    schema = solve_relation_parameter_schema()
    regions = least_squares_solver_coverage_regions()
    region = schema.covering_region(descriptor, regions)
    if region is None:
        raise ValueError(f"no least-squares solver covers descriptor {descriptor!r}")

    return region.owner


def select_root_solver_for_descriptor(
    descriptor: ParameterDescriptor,
) -> type:
    """Select a root solver by primitive solve-relation descriptor coverage."""
    schema = solve_relation_parameter_schema()
    regions = root_solver_coverage_regions()
    region = schema.covering_region(descriptor, regions)
    if region is None:
        raise ValueError(f"no root solver covers descriptor {descriptor!r}")

    return region.owner


def select_spectral_solver_for_descriptor(
    descriptor: ParameterDescriptor,
) -> type:
    """Select a spectral solver by primitive solve-relation coverage."""
    schema = solve_relation_parameter_schema()
    regions = spectral_solver_coverage_regions()
    region = schema.covering_region(descriptor, regions)
    if region is None:
        raise ValueError(f"no spectral solver covers descriptor {descriptor!r}")

    return region.owner


__all__ = [
    "linear_solver_coverage_regions",
    "LINEAR_SOLVER_COVERAGE_REGIONS",
    "least_squares_solver_coverage_regions",
    "LEAST_SQUARES_SOLVER_COVERAGE_REGIONS",
    "root_solver_coverage_regions",
    "ROOT_SOLVER_COVERAGE_REGIONS",
    "spectral_solver_coverage_regions",
    "SPECTRAL_SOLVER_COVERAGE_REGIONS",
    "select_least_squares_solver_for_descriptor",
    "select_linear_solver_for_descriptor",
    "select_root_solver_for_descriptor",
    "select_spectral_solver_for_descriptor",
]
