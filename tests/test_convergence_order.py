"""Parametric convergence order verification for all registered convergent classes.

Every concrete subclass of a convergent ABC (NumericalFlux, and future
DiscreteOperator, LinearSolver, ...) must have an oracle registered in
tests/support/oracles/.  conftest.py enforces this at collection time and
parametrizes this test over every (oracle, instance) pair.
"""

from __future__ import annotations

import sympy


def test_convergence_order(convergence_case: tuple) -> None:
    oracle, instance = convergence_case
    h = sympy.Symbol("h", positive=True)
    error = sympy.expand(sympy.simplify(oracle.error(instance, h)))
    poly = sympy.Poly(error, h)
    for k in range(instance.order):
        assert poly.nth(k) == 0, (
            f"Unexpected O(h^{k}) term in {type(instance).__name__}"
            f"(order={instance.order}): {poly.nth(k)}"
        )
    assert poly.nth(instance.order) != 0, (
        f"Missing O(h^{instance.order}) leading term in "
        f"{type(instance).__name__}(order={instance.order})"
    )
