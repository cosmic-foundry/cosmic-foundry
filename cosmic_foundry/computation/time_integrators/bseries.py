"""B-series rooted-tree framework for Runge-Kutta order verification.

Rooted trees are represented as sorted tuples of child trees (recursively).
The leaf (single node) is the empty tuple ().

    ()        — leaf, order 1
    ((),)     — root with one leaf child, order 2
    ((), ())  — root with two leaf children, order 3
    (((),),)  — root with one 2-node child, order 3

The B-series order conditions for an explicit RK method with Butcher tableau
(A, b) are: α(τ) = 1/γ(τ) for all rooted trees τ with |τ| ≤ p.

Counts of trees by order: 1, 1, 2, 4, 9, 20, 48, ...
"""

from __future__ import annotations

from collections import Counter
from functools import cache
from math import factorial

import sympy

Tree = tuple  # Recursive: tuple["Tree", ...]


@cache
def order(t: Tree) -> int:
    """Number of nodes |τ|."""
    return 1 + sum(order(c) for c in t)


@cache
def gamma(t: Tree) -> sympy.Rational:
    """Tree density γ(τ) = |τ| · ∏ γ(τᵢ) over children τᵢ."""
    result = sympy.Rational(order(t))
    for c in t:
        result *= gamma(c)
    return result


@cache
def sigma(t: Tree) -> sympy.Rational:
    """Symmetry factor σ(τ) = ∏ mᵢ! · σ(τᵢ)^mᵢ for repeated children."""
    result = sympy.Rational(1)
    for child, m in Counter(t).items():
        result *= factorial(m) * sigma(child) ** m
    return result


def trees_up_to_order(p: int) -> list[Tree]:
    """All rooted trees τ with 1 ≤ |τ| ≤ p, listed in increasing order of |τ|."""
    by_order: dict[int, list[Tree]] = {}
    for n in range(1, p + 1):
        by_order[n] = _trees_of_order(n, by_order)
    return [t for n in range(1, p + 1) for t in by_order[n]]


def elementary_weight(
    t: Tree,
    A: list[list[sympy.Rational]],
    b: list[sympy.Rational],
) -> sympy.Rational:
    """Elementary weight α(τ) = bᵀ e(τ) for an explicit RK method.

    Stage weights e(τ) ∈ ℝˢ follow the Butcher recursion:

        e(•)ᵢ = 1
        e(τ)ᵢ = ∏_{child τₗ of τ} (∑ⱼ Aᵢⱼ e(τₗ)ⱼ)    for |τ| > 1
    """
    e = _stage_weights(t, A)
    s = len(b)
    return sum((b[i] * e[i] for i in range(s)), sympy.Rational(0))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _trees_of_order(n: int, by_order: dict[int, list[Tree]]) -> list[Tree]:
    if n == 1:
        return [()]
    return _sorted_forests(n - 1, by_order, ())


def _sorted_forests(
    remaining: int,
    by_order: dict[int, list[Tree]],
    min_child: Tree,
) -> list[Tree]:
    """Sorted tuples of trees summing to `remaining` nodes, all children ≥ min_child."""
    if remaining == 0:
        return [()]
    result: list[Tree] = []
    for n1 in range(1, remaining + 1):
        for t1 in by_order.get(n1, []):
            if t1 < min_child:
                continue
            for rest in _sorted_forests(remaining - n1, by_order, t1):
                result.append((t1,) + rest)
    return result


def _stage_weights(
    t: Tree,
    A: list[list[sympy.Rational]],
) -> list[sympy.Rational]:
    s = len(A)
    if len(t) == 0:  # leaf
        return [sympy.Rational(1)] * s
    child_Ae: list[list[sympy.Rational]] = []
    for child in t:
        e_child = _stage_weights(child, A)
        child_Ae.append(
            [
                sum((A[i][j] * e_child[j] for j in range(s)), sympy.Rational(0))
                for i in range(s)
            ]
        )
    result: list[sympy.Rational] = []
    for i in range(s):
        val = sympy.Rational(1)
        for Ae in child_Ae:
            val *= Ae[i]
        result.append(val)
    return result


__all__ = [
    "Tree",
    "elementary_weight",
    "gamma",
    "order",
    "sigma",
    "trees_up_to_order",
]
