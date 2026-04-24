"""Enforce that every subclass of a generic class specifies type parameters.

A class written as `class Foo(Bar):` where Bar is generic leaves Bar's
TypeVars unbound, silently falling back to `Any` in mypy's analysis.
mypy's --disallow-any-generics flag does not catch this in class
definitions, only in annotations.  This test catches it at the structural
level by inspecting __orig_bases__ at runtime.

Detection logic: a base appearing in __orig_bases__ as a plain `type`
(not a _GenericAlias) with non-empty __parameters__ is a generic class
used without type arguments.  Parameterized bases like Function[Field, Field]
appear as _GenericAlias objects, not plain types, so they pass.
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
        for base in getattr(cls, "__orig_bases__", ()):
            if isinstance(base, type) and getattr(base, "__parameters__", ()):
                violations.append(
                    f"{modname}.{clsname}: "
                    f"base '{base.__name__}' is generic but unparameterized"
                )
    if violations:
        raise AssertionError(
            "Classes with unparameterized generic bases:\n"
            + "\n".join(f"  {v}" for v in violations)
        )
