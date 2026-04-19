"""Enforce that theory/ contains no third-party imports.

The abstract-to-concrete boundary in this codebase is defined precisely
by the third-party import boundary: theory/ may only import from the
Python standard library or from within cosmic_foundry.theory itself.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

THEORY_DIR = Path(__file__).parent.parent / "cosmic_foundry" / "theory"
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


def test_theory_has_no_third_party_imports() -> None:
    failures: dict[str, list[str]] = {}
    for path in sorted(THEORY_DIR.rglob("*.py")):
        violations = _third_party_imports(path)
        if violations:
            failures[str(path.relative_to(THEORY_DIR))] = violations

    if failures:
        lines = ["theory/ must not import third-party packages:"]
        for file, imports in failures.items():
            lines.append(f"  {file}: {', '.join(imports)}")
        raise AssertionError("\n".join(lines))
