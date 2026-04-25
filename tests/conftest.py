"""Shared pytest fixtures and configuration."""

import pytest

from tests.support.convergence_registry import check_registry_complete, iter_instances


@pytest.fixture(autouse=True, scope="session")
def _check_registry_complete() -> None:
    check_registry_complete()


def pytest_generate_tests(metafunc):  # type: ignore[no-untyped-def]
    if "convergence_case" in metafunc.fixturenames:
        cases = list(iter_instances())
        ids = [f"{type(i).__name__}(order={i.order})" for i in cases]
        metafunc.parametrize("convergence_case", cases, ids=ids)
