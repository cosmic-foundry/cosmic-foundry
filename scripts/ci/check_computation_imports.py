#!/usr/bin/env python3
"""Enforce that cosmic_foundry/computation/ does not import from other
cosmic_foundry sub-packages.

Files inside computation/ may only import from:
  - the standard library and third-party packages (numpy, math, …)
  - cosmic_foundry.computation itself

Importing from cosmic_foundry.geometry, cosmic_foundry.theory, or any
other cosmic_foundry sub-package is a layering violation: computation/
is the bottom-most numeric layer and must remain mesh-agnostic.

Uses the AST to detect imports, so strings and comments are ignored.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LIBRARY_ROOT = REPO_ROOT / "cosmic_foundry"
COMPUTATION_ROOT = LIBRARY_ROOT / "computation"
SKIP_DIRS = {"miniforge", ".git"}

# The only cosmic_foundry sub-package that computation/ may import from.
_ALLOWED_CF_PREFIX = "cosmic_foundry.computation"


def _layering_violations(path: Path) -> list[tuple[int, str]]:
    """Return (lineno, module) for every intra-package layering violation."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError:
        return []
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(
                    "cosmic_foundry."
                ) and not alias.name.startswith(_ALLOWED_CF_PREFIX):
                    hits.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith("cosmic_foundry."):
                if not node.module.startswith(_ALLOWED_CF_PREFIX):
                    hits.append((node.lineno, node.module))
    return hits


def scan() -> list[str]:
    errors: list[str] = []
    for path in sorted(COMPUTATION_ROOT.rglob("*.py")):
        if SKIP_DIRS.intersection(path.parts):
            continue
        for lineno, module in _layering_violations(path):
            rel = path.relative_to(REPO_ROOT)
            errors.append(
                f"{rel}:{lineno}: computation/ imports from outside layer '{module}'"
            )
    return errors


def main() -> int:
    errors = scan()
    for err in errors:
        print(err, file=sys.stderr)
    if errors:
        print(
            f"\n{len(errors)} violation(s) — computation/ must not import from other"
            " cosmic_foundry packages",
            file=sys.stderr,
        )
        return 1
    print("No layering violations in cosmic_foundry/computation/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
