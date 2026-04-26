"""Auto-parametrized structural tests.

Auto-discovered over every ABC and module in the codebase:
  1. Every ABC cannot be directly instantiated (abstract-method guard).
  2. Every cosmic_foundry ancestor in the MRO is a proper superclass.
  3. Every class defined in a module appears in that module's __all__.
  4. Every concrete IterativeSolver and Factorization processes declared
     Tensors without triggering MaterializationError (JIT-compatibility).

Discovery runs at import time, so any new ABC or module is covered without
touching this file.
"""

from __future__ import annotations

import importlib
import inspect
import types
from abc import ABC
from pathlib import Path
from typing import Any

import pytest

from cosmic_foundry.computation.factorization import Factorization
from cosmic_foundry.computation.iterative_solver import IterativeSolver
from cosmic_foundry.computation.tensor import MaterializationError, Tensor

_PROJECT_ROOT = Path(__file__).parent.parent
_PACKAGES = [
    "cosmic_foundry.theory.foundation",
    "cosmic_foundry.theory.continuous",
    "cosmic_foundry.theory.discrete",
    "cosmic_foundry.geometry",
    "cosmic_foundry.physics",
    "cosmic_foundry.computation",
]

_JIT_N = 4  # standard declared-tensor size used for JIT-compat checks


def _discover_modules() -> list[tuple[str, types.ModuleType]]:
    result = []
    for pkg in _PACKAGES:
        pkg_path = _PROJECT_ROOT / pkg.replace(".", "/")
        for path in sorted(pkg_path.glob("*.py")):
            if path.stem == "__init__":
                continue
            mod_path = f"{pkg}.{path.stem}"
            try:
                mod = importlib.import_module(mod_path)
            except ImportError:
                continue
            result.append((mod_path, mod))
    return result


def _discover_abcs() -> list[type]:
    seen: set[type] = set()
    abcs: list[type] = []
    for mod_path, mod in _MODULES:
        for _, obj in inspect.getmembers(mod, inspect.isclass):
            if (
                obj.__module__ == mod_path
                and getattr(obj, "__abstractmethods__", None)
                and obj not in seen
            ):
                seen.add(obj)
                abcs.append(obj)
    return abcs


def _discover_hierarchy_pairs() -> list[tuple[type, type]]:
    seen: set[tuple[type, type]] = set()
    pairs: list[tuple[type, type]] = []
    for cls in _ABCS:
        for base in inspect.getmro(cls)[1:]:
            if base is object:
                continue
            if not getattr(base, "__module__", "").startswith("cosmic_foundry"):
                continue
            pair = (cls, base)
            if pair not in seen:
                seen.add(pair)
                pairs.append(pair)
    return pairs


def _is_concrete(cls: type) -> bool:
    return (
        inspect.isclass(cls)
        and not getattr(cls, "__abstractmethods__", None)
        and not issubclass(cls, ABC)
        or (inspect.isclass(cls) and not getattr(cls, "__abstractmethods__", None))
    )


def _discover_concrete_iterative_solvers() -> list[type]:
    seen: set[type] = set()
    result: list[type] = []
    for _, mod in _MODULES:
        for _, obj in inspect.getmembers(mod, inspect.isclass):
            if (
                obj not in seen
                and issubclass(obj, IterativeSolver)
                and not getattr(obj, "__abstractmethods__", None)
                and obj is not IterativeSolver
            ):
                seen.add(obj)
                result.append(obj)
    return result


def _discover_concrete_factorizations() -> list[type]:
    seen: set[type] = set()
    result: list[type] = []
    for _, mod in _MODULES:
        for _, obj in inspect.getmembers(mod, inspect.isclass):
            if (
                obj not in seen
                and issubclass(obj, Factorization)
                and not getattr(obj, "__abstractmethods__", None)
                and obj is not Factorization
            ):
                seen.add(obj)
                result.append(obj)
    return result


# ---------------------------------------------------------------------------
# JIT-compatibility claims
# ---------------------------------------------------------------------------


class _IterativeJitClaim:
    """Claim: IterativeSolver.init_state/step/converged run on declared Tensors."""

    def __init__(self, cls: type) -> None:
        self._cls = cls

    @property
    def description(self) -> str:
        return self._cls.__qualname__

    def check(self) -> None:
        n = _JIT_N
        a = Tensor.declare(n, n)
        b = Tensor.declare(n)
        solver: Any = self._cls()
        state = solver.init_state(a, b)
        new_state = solver.step(state)
        assert type(new_state) is type(state)
        converged = solver.converged(state)
        assert isinstance(converged, Tensor)
        assert converged.shape == ()

    def check_materialization_gate(self) -> None:
        """converged() returns an unallocated Tensor; .get() raises as intended."""
        n = _JIT_N
        a = Tensor.declare(n, n)
        b = Tensor.declare(n)
        solver: Any = self._cls()
        state = solver.init_state(a, b)
        converged = solver.converged(state)
        with pytest.raises(MaterializationError):
            converged.get()


class _FactorizationJitClaim:
    """Claim: Factorization.factorize / FactoredMatrix.solve run on declared Tensors."""

    def __init__(self, cls: type) -> None:
        self._cls = cls

    @property
    def description(self) -> str:
        return self._cls.__qualname__

    def check(self) -> None:
        n = _JIT_N
        a = Tensor.declare(n, n)
        rhs = Tensor.declare(n)
        factored = self._cls().factorize(a)
        factored.solve(rhs)


_MODULES = _discover_modules()
_ABCS = _discover_abcs()
_HIERARCHY_PAIRS = _discover_hierarchy_pairs()

_ITERATIVE_JIT_CLAIMS = [
    _IterativeJitClaim(cls) for cls in _discover_concrete_iterative_solvers()
]
_FACTORIZATION_JIT_CLAIMS = [
    _FactorizationJitClaim(cls) for cls in _discover_concrete_factorizations()
]


# ---------------------------------------------------------------------------
# Structural tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", _ABCS, ids=lambda c: c.__qualname__)
def test_abc_is_not_directly_instantiable(cls: type) -> None:
    with pytest.raises(TypeError):
        cls()


@pytest.mark.parametrize(
    "child,parent",
    _HIERARCHY_PAIRS,
    ids=[f"{c.__qualname__}->{p.__qualname__}" for c, p in _HIERARCHY_PAIRS],
)
def test_hierarchy_issubclass(child: type, parent: type) -> None:
    assert issubclass(child, parent)


@pytest.mark.parametrize("mod_path,mod", _MODULES, ids=[m for m, _ in _MODULES])
def test_module_all_is_complete(mod_path: str, mod: types.ModuleType) -> None:
    exported = set(getattr(mod, "__all__", []))
    defined = {
        name
        for name, obj in inspect.getmembers(mod, inspect.isclass)
        if obj.__module__ == mod_path and not name.startswith("_")
    }
    missing = defined - exported
    assert not missing, f"defined but not in __all__: {missing}"


# ---------------------------------------------------------------------------
# JIT-compatibility tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("claim", _ITERATIVE_JIT_CLAIMS, ids=lambda c: c.description)
def test_iterative_solver_jit_compatible(claim: _IterativeJitClaim) -> None:
    claim.check()


@pytest.mark.parametrize("claim", _ITERATIVE_JIT_CLAIMS, ids=lambda c: c.description)
def test_iterative_solver_materialization_gate(claim: _IterativeJitClaim) -> None:
    claim.check_materialization_gate()


@pytest.mark.parametrize(
    "claim", _FACTORIZATION_JIT_CLAIMS, ids=lambda c: c.description
)
def test_factorization_jit_compatible(claim: _FactorizationJitClaim) -> None:
    claim.check()
