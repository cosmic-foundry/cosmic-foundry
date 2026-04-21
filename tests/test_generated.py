"""Auto-parametrized structural and derived-property tests.

Structural invariants (auto-discovered over every ABC in the codebase):
  1. Every ABC cannot be directly instantiated (abstract-method guard).
  2. Every cosmic_foundry ancestor in the MRO is a proper superclass.

Derived-property invariants (per-ABC mathematical laws verified on stubs):
  3. DifferentialForm.tensor_type == (0, degree) for degree in 0..3.
  4. PseudoRiemannianManifold.ndim == sum(signature) for several signatures.
  5. SymmetricTensorField.component(i, j) == component(j, i) pointwise.

Discovery for (1) and (2) runs at import time, so any new ABC is covered
without touching the test suite.
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from typing import Any

import pytest

from cosmic_foundry.continuous.differential_form import DifferentialForm
from cosmic_foundry.continuous.field import Field, SymmetricTensorField
from cosmic_foundry.continuous.manifold import Manifold
from cosmic_foundry.continuous.pseudo_riemannian_manifold import (
    PseudoRiemannianManifold,
)

_PROJECT_ROOT = Path(__file__).parent.parent
_PACKAGES = [
    "cosmic_foundry.foundation",
    "cosmic_foundry.continuous",
    "cosmic_foundry.discrete",
]


def _discover_abcs() -> list[type]:
    seen: set[type] = set()
    abcs: list[type] = []
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
    for cls in _discover_abcs():
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


_ABCS = _discover_abcs()
_HIERARCHY_PAIRS = _discover_hierarchy_pairs()


@pytest.mark.parametrize("cls", _ABCS, ids=lambda c: c.__qualname__)
def test_abc_is_not_directly_instantiable(cls: type) -> None:
    with pytest.raises(TypeError):
        cls()  # type: ignore[abstract]


@pytest.mark.parametrize(
    "child,parent",
    _HIERARCHY_PAIRS,
    ids=[f"{c.__qualname__}->{p.__qualname__}" for c, p in _HIERARCHY_PAIRS],
)
def test_hierarchy_issubclass(child: type, parent: type) -> None:
    assert issubclass(child, parent)


# ---------------------------------------------------------------------------
# Derived-property invariant stubs
# ---------------------------------------------------------------------------


class _ManifoldStub(Manifold):
    @property
    def ndim(self) -> int:
        return 3

    @property
    def atlas(self) -> Any:
        raise NotImplementedError


_M = _ManifoldStub()


class _DFStub(DifferentialForm):
    def __init__(self, degree: int) -> None:
        self._degree = degree

    @property
    def degree(self) -> int:
        return self._degree

    @property
    def manifold(self) -> Manifold:
        return _M

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        return None


class _PRMStub(PseudoRiemannianManifold):
    def __init__(self, p: int, q: int) -> None:
        self._sig = (p, q)

    @property
    def signature(self) -> tuple[int, int]:
        return self._sig

    @property
    def metric(self) -> Any:
        raise NotImplementedError

    @property
    def atlas(self) -> Any:
        raise NotImplementedError


class _SymStub(SymmetricTensorField):
    @property
    def manifold(self) -> Manifold:
        return _M

    def component(self, i: int, j: int) -> Field:
        val = float(min(i, j) + max(i, j) * 10)

        class _C(Field):
            @property
            def manifold(self) -> Manifold:
                return _M

            def __call__(self, *a: Any, **kw: Any) -> float:
                return val

        return _C()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return None


# ---------------------------------------------------------------------------
# Derived-property invariant tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", range(4))
def test_differential_form_tensor_type_derived(k: int) -> None:
    assert _DFStub(k).tensor_type == (0, k)


@pytest.mark.parametrize("signature", [(1, 3), (3, 0), (2, 2), (0, 4)])
def test_pseudo_riemannian_ndim_derived(signature: tuple[int, int]) -> None:
    p, q = signature
    assert _PRMStub(p, q).ndim == p + q


@pytest.mark.parametrize("i,j", [(0, 1), (0, 2), (1, 2)])
def test_symmetric_tensor_component_symmetry(i: int, j: int) -> None:
    stub = _SymStub()
    assert stub.component(i, j)(None) == stub.component(j, i)(None)
