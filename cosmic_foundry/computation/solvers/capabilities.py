"""Autodiscovered linear-solver coverage aggregation."""

from __future__ import annotations

from importlib import import_module
from pkgutil import iter_modules
from types import ModuleType

from cosmic_foundry.computation.algorithm_capabilities import (
    CoverageRegion,
    ParameterDescriptor,
    linear_solver_parameter_schema,
)
from cosmic_foundry.computation.solvers.coverage import coverage


def _solver_package_modules() -> tuple[ModuleType, ...]:
    package = import_module(__package__ or "cosmic_foundry.computation.solvers")
    package_path = getattr(package, "__path__", ())
    modules: list[ModuleType] = []
    for module_info in sorted(iter_modules(package_path), key=lambda info: info.name):
        if module_info.name.startswith("_") or module_info.name == "capabilities":
            continue
        modules.append(import_module(f"{package.__name__}.{module_info.name}"))
    return tuple(modules)


def _discovered_coverage_regions() -> tuple[CoverageRegion, ...]:
    regions: list[CoverageRegion] = []
    for module in _solver_package_modules():
        for item in module.__dict__.values():
            if not isinstance(item, type) or item.__module__ != module.__name__:
                continue
            predicates = getattr(item, "linear_solver_coverage", None)
            if predicates is not None:
                regions.append(
                    coverage(
                        item,
                        coverage_predicates=predicates,
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
