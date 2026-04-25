"""Shared pytest fixtures and configuration."""

import tests.support.instances  # noqa: F401 — registers all convergence instances
from tests.support.convergence_registry import check_registry_complete, iter_instances

check_registry_complete()


def pytest_generate_tests(metafunc):  # type: ignore[no-untyped-def]
    if "convergence_case" in metafunc.fixturenames:
        cases = list(iter_instances())
        ids = [
            f"{type(instance).__name__}(order={instance.order})" for instance in cases
        ]
        metafunc.parametrize("convergence_case", cases, ids=ids)
