"""Autodiscovered linear-solver capability aggregation."""

from __future__ import annotations

from importlib import import_module
from pkgutil import iter_modules
from types import ModuleType

from cosmic_foundry.computation.algorithm_capabilities import (
    AlgorithmCapability,
    AlgorithmRegistry,
    CoveragePatch,
    ParameterDescriptor,
    linear_solver_parameter_schema,
)
from cosmic_foundry.computation.solvers._capability_claims import (
    selector_rejection_patches,
)

LinearSolverCapability = AlgorithmCapability
LinearSolverRegistry = AlgorithmRegistry


def _solver_package_modules() -> tuple[ModuleType, ...]:
    package = import_module(__package__ or "cosmic_foundry.computation.solvers")
    package_path = getattr(package, "__path__", ())
    modules: list[ModuleType] = []
    for module_info in sorted(iter_modules(package_path), key=lambda info: info.name):
        if module_info.name.startswith("_") or module_info.name == "capabilities":
            continue
        modules.append(import_module(f"{package.__name__}.{module_info.name}"))
    return tuple(modules)


def _declared_capabilities() -> tuple[LinearSolverCapability, ...]:
    capabilities: list[LinearSolverCapability] = []
    for module in _solver_package_modules():
        for item in module.__dict__.values():
            if not isinstance(item, type) or item.__module__ != module.__name__:
                continue
            declare = getattr(item, "linear_solver_capabilities", None)
            if declare is not None:
                capabilities.extend(declare())
    return tuple(capabilities)


LINEAR_SOLVER_REGISTRY = LinearSolverRegistry(_declared_capabilities())


def linear_solver_capabilities() -> tuple[LinearSolverCapability, ...]:
    """Return autodiscovered linear-solver algorithm capabilities."""
    return LINEAR_SOLVER_REGISTRY.capabilities


def linear_solver_coverage_patches() -> tuple[CoveragePatch, ...]:
    """Return autodiscovered descriptor-space coverage patches."""
    patches: list[CoveragePatch] = []
    for capability in LINEAR_SOLVER_REGISTRY.capabilities:
        patches.extend(capability.coverage_patches)
    patches.extend(selector_rejection_patches())
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
    "LINEAR_SOLVER_REGISTRY",
    "select_linear_solver_for_descriptor",
]
