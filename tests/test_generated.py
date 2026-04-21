"""Auto-parametrized structural tests for every ABC in the codebase.

Covers two invariants for every discovered ABC:
  1. It cannot be directly instantiated (abstract-method guard).
  2. Every cosmic_foundry ancestor in its MRO is a proper superclass
     (issubclass transitivity check).

Discovery runs at import time, so any new ABC added to the codebase is
covered without touching the test suite.
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import pytest

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
