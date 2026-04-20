"""Enforce that foundation/ and continuous/ contain no third-party imports.

The abstract-to-concrete boundary in this codebase is defined precisely
by the third-party import boundary: foundation/ and continuous/ may only
import from the Python standard library or from within cosmic_foundry.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).parent.parent / "cosmic_foundry"
PURE_PACKAGES = [
    PACKAGE_ROOT / "foundation",
    PACKAGE_ROOT / "continuous",
]
STDLIB = sys.stdlib_module_names


def _top_level(module: str) -> str:
    return module.split(".")[0]


def _third_party_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = _top_level(alias.name)
                if top not in STDLIB and top != "cosmic_foundry":
                    violations.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                continue
            top = _top_level(node.module)
            if top not in STDLIB and top != "cosmic_foundry":
                violations.append(node.module)
    return violations


def test_pure_packages_have_no_third_party_imports() -> None:
    failures: dict[str, list[str]] = {}
    for package_dir in PURE_PACKAGES:
        for path in sorted(package_dir.rglob("*.py")):
            violations = _third_party_imports(path)
            if violations:
                rel = path.relative_to(PACKAGE_ROOT.parent)
                failures[str(rel)] = violations

    if failures:
        lines = ["foundation/ and continuous/ must not import third-party packages:"]
        for file, imports in failures.items():
            lines.append(f"  {file}: {', '.join(imports)}")
        raise AssertionError("\n".join(lines))
