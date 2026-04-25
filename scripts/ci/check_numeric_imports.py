#!/usr/bin/env python3
"""Enforce that numeric computation libraries are only imported inside
cosmic_foundry/computation/.

The libraries math, numpy, scipy, jax, and torch must not be imported
anywhere in cosmic_foundry/ outside the computation sub-package.  All
numeric operations on Tensor data belong in that layer; higher-level
layers (theory, geometry, …) interact with numbers exclusively through
the Tensor API.

Uses the AST to detect imports, so strings and comments are ignored.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

BANNED = {"math", "numpy", "scipy", "jax", "torch"}
REPO_ROOT = Path(__file__).resolve().parents[2]
LIBRARY_ROOT = REPO_ROOT / "cosmic_foundry"
ALLOWED_PREFIX = LIBRARY_ROOT / "computation"
SKIP_DIRS = {"miniforge", ".git"}


def _banned_imports(path: Path) -> list[tuple[int, str]]:
    """Return (lineno, module) for every banned import in path."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError:
        return []
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in BANNED:
                    hits.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root = node.module.split(".")[0]
                if root in BANNED:
                    hits.append((node.lineno, node.module))
    return hits


def scan() -> list[str]:
    errors: list[str] = []
    for path in sorted(LIBRARY_ROOT.rglob("*.py")):
        if SKIP_DIRS.intersection(path.parts):
            continue
        try:
            path.relative_to(ALLOWED_PREFIX)
            continue  # inside computation/ — allowed
        except ValueError:
            pass
        for lineno, module in _banned_imports(path):
            rel = path.relative_to(REPO_ROOT)
            errors.append(f"{rel}:{lineno}: banned numeric import '{module}'")
    return errors


def main() -> int:
    errors = scan()
    for err in errors:
        print(err, file=sys.stderr)
    if errors:
        print(
            f"\n{len(errors)} violation(s) — numeric imports outside computation/",
            file=sys.stderr,
        )
        return 1
    print("No numeric imports outside cosmic_foundry/computation/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
