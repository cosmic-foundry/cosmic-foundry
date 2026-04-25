"""Shared pytest fixtures and configuration."""

import tests.support.oracles  # noqa: F401 — registers all convergence oracles
from tests.support.convergence_registry import check_registry_complete, iter_cases

check_registry_complete()


def pytest_generate_tests(metafunc):
    if "convergence_case" in metafunc.fixturenames:
        cases = list(iter_cases())
        ids = [
            f"{type(instance).__name__}(order={instance.order})"
            for _, instance in cases
        ]
        metafunc.parametrize("convergence_case", cases, ids=ids)
