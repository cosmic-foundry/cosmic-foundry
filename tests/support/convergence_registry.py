"""Generic convergence registry.

CONVERGENT_ABCS: ABCs whose every concrete subclass must have instances
registered before tests run.  Add a new ABC when a new convergent hierarchy
is introduced (e.g. LinearSolver).

CONVERGENCE_INSTANCES: maps each concrete class to a list of test instances.
Populated as a side-effect of importing the relevant conftest (e.g.
tests/geometry/conftest.py registers DiscreteOperator subclasses).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from cosmic_foundry.theory.discrete.discrete_operator import DiscreteOperator

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
    """Assert every concrete convergent class has instances registered."""
    for abc in CONVERGENT_ABCS:
        for cls in _all_concrete_subclasses(abc):
            assert cls in CONVERGENCE_INSTANCES, (
                f"{cls.__qualname__} is a concrete subclass of "
                f"{abc.__qualname__} but has no instances registered. "
                f"Add entries in the relevant tests/*/conftest.py."
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
