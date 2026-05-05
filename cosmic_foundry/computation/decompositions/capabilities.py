"""Autodiscovered decomposition coverage aggregation."""

from __future__ import annotations

from importlib import import_module
from pkgutil import iter_modules
from types import ModuleType

from cosmic_foundry.computation.algorithm_capabilities import (
    AffineComparisonPredicate,
    CoverageRegion,
    DecompositionField,
    ParameterDescriptor,
    StructuredPredicate,
    decomposition_parameter_schema,
)
from cosmic_foundry.computation.decompositions.decomposition import Decomposition
from cosmic_foundry.computation.decompositions.factorization import Factorization


def _decomposition_package_modules() -> tuple[ModuleType, ...]:
    package = import_module(__package__ or "cosmic_foundry.computation.decompositions")
    package_path = getattr(package, "__path__", ())
    modules: list[ModuleType] = []
    for module_info in sorted(iter_modules(package_path), key=lambda info: info.name):
        if module_info.name.startswith("_") or module_info.name == "capabilities":
            continue
        modules.append(import_module(f"{package.__name__}.{module_info.name}"))
    return tuple(modules)


def _budget_predicates() -> tuple[AffineComparisonPredicate, ...]:
    return (
        AffineComparisonPredicate(
            {
                DecompositionField.FACTORIZATION_WORK_BUDGET_FMAS: 1.0,
                DecompositionField.FACTORIZATION_WORK_FMAS: -1.0,
            },
            ">=",
            0.0,
        ),
        AffineComparisonPredicate(
            {
                DecompositionField.FACTORIZATION_MEMORY_BUDGET_BYTES: 1.0,
                DecompositionField.FACTORIZATION_MEMORY_BYTES: -1.0,
            },
            ">=",
            0.0,
        ),
    )


def _inherited_coverage_regions(
    owner: type,
) -> tuple[tuple[StructuredPredicate, ...], ...]:
    prefixes: tuple[StructuredPredicate, ...] = ()
    if issubclass(owner, Factorization):
        prefixes += _budget_predicates()
    regions = getattr(owner, "factorization_feasibility_regions", ())
    return tuple(prefixes + region for region in regions)


def _discovered_coverage_regions() -> tuple[CoverageRegion, ...]:
    regions: list[CoverageRegion] = []
    for module in _decomposition_package_modules():
        for item in module.__dict__.values():
            if not isinstance(item, type) or item.__module__ != module.__name__:
                continue
            if (
                issubclass(item, Decomposition)
                and not getattr(item, "__abstractmethods__", None)
                and item is not Decomposition
            ):
                regions.extend(
                    CoverageRegion(item, predicates)
                    for predicates in _inherited_coverage_regions(item)
                )
    return tuple(regions)


DECOMPOSITION_COVERAGE_REGIONS = _discovered_coverage_regions()


def decomposition_coverage_regions() -> tuple[CoverageRegion, ...]:
    """Return autodiscovered descriptor-space decomposition coverage regions."""
    return DECOMPOSITION_COVERAGE_REGIONS


def select_decomposition_for_descriptor(descriptor: ParameterDescriptor) -> type:
    """Select a decomposition by parameter-space descriptor coverage."""
    schema = decomposition_parameter_schema()
    regions = decomposition_coverage_regions()
    region = schema.covering_region(descriptor, regions)
    if region is None:
        raise ValueError(f"no decomposition covers descriptor {descriptor!r}")
    return region.owner


__all__ = [
    "DECOMPOSITION_COVERAGE_REGIONS",
    "decomposition_coverage_regions",
    "select_decomposition_for_descriptor",
]
