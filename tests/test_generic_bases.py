"""Enforce that every subclass of a generic class specifies type parameters.

A class written as `class Foo(Bar):` where Bar is generic leaves Bar's
TypeVars unbound, silently falling back to `Any` in mypy's analysis.
mypy's --disallow-any-generics flag does not catch this in class
definitions, only in annotations.  This test catches it at the structural
level by inspecting __orig_bases__ at runtime.

Detection logic: a class violates if its own __parameters__ is empty
(it is not itself generic) but any base in __orig_bases__ has non-empty
__parameters__ (the base still carries free TypeVars).  Classes that are
themselves generic — i.e., they re-expose TypeVars for callers to bind —
have non-empty __parameters__ and are excluded from the check.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil

import cosmic_foundry


def _all_local_classes() -> list[tuple[str, str, type]]:
    """Return (modname, clsname, cls) for every class defined in cosmic_foundry."""
    results = []
    for _, modname, _ in pkgutil.walk_packages(
        cosmic_foundry.__path__,
        prefix="cosmic_foundry.",
    ):
        try:
            module = importlib.import_module(modname)
        except ImportError:
            continue
        for clsname, cls in inspect.getmembers(module, inspect.isclass):
            if cls.__module__ == modname:
                results.append((modname, clsname, cls))
    return results


def test_all_generic_bases_are_parameterized() -> None:
    violations: list[str] = []
    for modname, clsname, cls in _all_local_classes():
        if getattr(cls, "__parameters__", ()):
            continue  # cls is itself generic — TypeVars intentionally free
        for base in getattr(cls, "__orig_bases__", ()):
            if getattr(base, "__parameters__", ()):
                violations.append(
                    f"{modname}.{clsname}: base '{base}' has unbound TypeVars"
                )
                break
    if violations:
        raise AssertionError(
            "Classes that inherit from generic bases without binding all TypeVars:\n"
            + "\n".join(f"  {v}" for v in violations)
        )
