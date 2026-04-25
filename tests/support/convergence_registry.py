"""Registry of convergent classes and their test instances.

CONVERGENT_ABCS is the list of abstract base classes whose concrete
subclasses must each have instances registered here before any tests run.
Add a new ABC to the list when a new convergent hierarchy is introduced
(e.g. DiscreteOperator, LinearSolver).

CONVERGENCE_INSTANCES maps each concrete class to a list of test instances.
Registration happens in tests/support/oracles/__init__.py as an import
side-effect so that all instances are loaded before check_registry_complete()
runs.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from cosmic_foundry.theory.discrete.discrete_operator import DiscreteOperator

# DiscreteOperator is the single convergence root: every concrete subclass
# must have instances registered.
CONVERGENT_ABCS: list[type] = [DiscreteOperator]

CONVERGENCE_INSTANCES: dict[type, list[Any]] = {}


def _all_concrete_subclasses(cls: type) -> list[type]:
    result = []
    for sub in cls.__subclasses__():
        if not getattr(sub, "__abstractmethods__", None):
            result.append(sub)
        result.extend(_all_concrete_subclasses(sub))
    return result


def check_registry_complete() -> None:
    """Assert every concrete convergent class has instances registered.

    Called at conftest load time so a missing registration is a
    collection-time error, not a silently untested class.
    """
    for abc in CONVERGENT_ABCS:
        for cls in _all_concrete_subclasses(abc):
            assert cls in CONVERGENCE_INSTANCES, (
                f"{cls.__qualname__} is a concrete subclass of "
                f"{abc.__qualname__} but has no instances registered. "
                f"Register some in tests/support/oracles/__init__.py."
            )


def iter_instances() -> Iterator[Any]:
    """Yield every registered test instance."""
    for instances in CONVERGENCE_INSTANCES.values():
        yield from instances


__all__ = [
    "CONVERGENT_ABCS",
    "CONVERGENCE_INSTANCES",
    "check_registry_complete",
    "iter_instances",
]
