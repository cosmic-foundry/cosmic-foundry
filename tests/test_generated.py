"""Auto-parametrized structural tests.

Auto-discovered over every ABC and module in the codebase:
  1. Every ABC cannot be directly instantiated (abstract-method guard).
  2. Every cosmic_foundry ancestor in the MRO is a proper superclass.
  3. Every class defined in a module appears in that module's __all__.

Discovery runs at import time, so any new ABC or module is covered without
touching this file.
"""

from __future__ import annotations

import importlib
import inspect
import types
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).parent.parent
_PACKAGES = [
    "cosmic_foundry.foundation",
    "cosmic_foundry.continuous",
]


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


_MODULES = _discover_modules()
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


@pytest.mark.parametrize("mod_path,mod", _MODULES, ids=[m for m, _ in _MODULES])
def test_module_all_is_complete(mod_path: str, mod: types.ModuleType) -> None:
    exported = set(getattr(mod, "__all__", []))
    defined = {
        name
        for name, obj in inspect.getmembers(mod, inspect.isclass)
        if obj.__module__ == mod_path
    }
    missing = defined - exported
    assert not missing, f"defined but not in __all__: {missing}"
