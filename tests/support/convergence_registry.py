"""Registry of convergence oracles for all concrete convergent classes.

CONVERGENT_ABCS is the list of abstract base classes whose concrete
subclasses must each have an oracle registered here before any tests run.
Add a new ABC to the list when a new convergent hierarchy is introduced
(e.g. DiscreteOperator, LinearSolver).

CONVERGENCE_ORACLES maps each concrete class to its oracle.  Registration
happens in tests/support/oracles/__init__.py as an import side-effect so
that all oracles are loaded before check_registry_complete() runs.
"""

from __future__ import annotations

from cosmic_foundry.theory.discrete.numerical_flux import NumericalFlux
from tests.support.convergence_oracle import ConvergenceOracle

# Extend this list when new convergent ABCs are introduced.
CONVERGENT_ABCS: list[type] = [NumericalFlux]

CONVERGENCE_ORACLES: dict[type, ConvergenceOracle] = {}  # type: ignore[type-arg]


def _all_concrete_subclasses(cls: type) -> list[type]:
    result = []
    for sub in cls.__subclasses__():
        if not getattr(sub, "__abstractmethods__", None):
            result.append(sub)
        result.extend(_all_concrete_subclasses(sub))
    return result


def check_registry_complete() -> None:
    """Assert every concrete convergent class has an oracle registered.

    Called at conftest load time so a missing oracle is a collection-time
    error, not a silently untested class.
    """
    for abc in CONVERGENT_ABCS:
        for cls in _all_concrete_subclasses(abc):
            assert cls in CONVERGENCE_ORACLES, (
                f"{cls.__qualname__} is a concrete subclass of "
                f"{abc.__qualname__} but has no convergence oracle. "
                f"Register one in tests/support/oracles/__init__.py."
            )


def iter_cases():  # type: ignore[return]
    """Yield (oracle, instance) for every registered (class, parameter) pair."""
    for oracle in CONVERGENCE_ORACLES.values():
        yield from ((oracle, instance) for instance in oracle.instances())


__all__ = [
    "CONVERGENT_ABCS",
    "CONVERGENCE_ORACLES",
    "check_registry_complete",
    "iter_cases",
]
