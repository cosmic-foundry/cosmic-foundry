#!/usr/bin/env python3
"""Shared AST import graph and dependency policies for CI checks."""
from __future__ import annotations

import ast
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LIBRARY_ROOT = REPO_ROOT / "cosmic_foundry"
COMPUTATION_ROOT = LIBRARY_ROOT / "computation"
TESTS_ROOT = REPO_ROOT / "tests"
SKIP_DIRS = {"miniforge", ".git", "__pycache__"}
NUMERIC_IMPORT_ROOTS = {"math", "numpy", "scipy", "jax", "torch"}
ALL_TESTS_TARGET = "tests"
DOC_ONLY_TEST_TARGET = "tests/test_structure.py"


@dataclass(frozen=True)
class ImportEdge:
    importer: str
    imported: str
    path: Path
    lineno: int


def _contains_skipped_dir(path: Path) -> bool:
    return bool(SKIP_DIRS.intersection(path.parts))


def module_for_path(path: Path, root: Path = REPO_ROOT) -> str | None:
    path = path.resolve()
    root = root.resolve()
    try:
        rel = path.relative_to(root)
    except ValueError:
        return None
    if path.suffix != ".py":
        return None
    parts = list(rel.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def path_for_module(module: str, root: Path = REPO_ROOT) -> Path | None:
    base = root / Path(*module.split("."))
    module_path = base.with_suffix(".py")
    if module_path.exists():
        return module_path
    package_path = base / "__init__.py"
    if package_path.exists():
        return package_path
    return None


def python_files(root: Path) -> list[Path]:
    return [
        path for path in sorted(root.rglob("*.py")) if not _contains_skipped_dir(path)
    ]


def parse_import_edges(path: Path) -> list[ImportEdge]:
    importer = module_for_path(path)
    if importer is None:
        return []
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError:
        return []

    edges: list[ImportEdge] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                edges.append(ImportEdge(importer, alias.name, path, node.lineno))
        elif isinstance(node, ast.ImportFrom) and node.module:
            edges.append(ImportEdge(importer, node.module, path, node.lineno))
            for alias in node.names:
                candidate = f"{node.module}.{alias.name}"
                if path_for_module(candidate) is not None:
                    edges.append(ImportEdge(importer, candidate, path, node.lineno))
    return edges


def internal_import_graph() -> dict[str, set[str]]:
    roots = [LIBRARY_ROOT, TESTS_ROOT]
    files = [path for root in roots for path in python_files(root)]
    graph = {module_for_path(path): set() for path in files}
    graph = {module: imports for module, imports in graph.items() if module is not None}
    for path in files:
        importer = module_for_path(path)
        if importer is None:
            continue
        for edge in parse_import_edges(path):
            if path_for_module(edge.imported) is not None:
                graph[importer].add(edge.imported)
    return graph


def reverse_reachable(graph: dict[str, set[str]], roots: set[str]) -> set[str]:
    reverse: dict[str, set[str]] = {}
    for importer, imports in graph.items():
        for imported in imports:
            reverse.setdefault(imported, set()).add(importer)

    seen = set(roots)
    frontier = list(roots)
    while frontier:
        module = frontier.pop()
        for importer in reverse.get(module, set()):
            if importer not in seen:
                seen.add(importer)
                frontier.append(importer)
    return seen


def computation_layer_violations() -> list[str]:
    errors: list[str] = []
    for path in python_files(COMPUTATION_ROOT):
        for edge in parse_import_edges(path):
            is_cosmic = edge.imported.startswith("cosmic_foundry.")
            is_computation = edge.imported.startswith("cosmic_foundry.computation")
            if is_cosmic and not is_computation:
                rel = edge.path.relative_to(REPO_ROOT)
                errors.append(
                    f"{rel}:{edge.lineno}: computation/ imports from outside "
                    f"layer '{edge.imported}'"
                )
    return errors


def numeric_import_violations() -> list[str]:
    errors: list[str] = []
    for path in python_files(LIBRARY_ROOT):
        try:
            path.relative_to(COMPUTATION_ROOT)
            continue
        except ValueError:
            pass
        for edge in parse_import_edges(path):
            root = edge.imported.split(".")[0]
            if root in NUMERIC_IMPORT_ROOTS:
                rel = edge.path.relative_to(REPO_ROOT)
                errors.append(
                    f"{rel}:{edge.lineno}: banned numeric import '{edge.imported}'"
                )
    return errors


def changed_files(base: str, head: str = "HEAD") -> list[Path]:
    proc = subprocess.run(
        ["git", "diff", "--name-only", f"{base}...{head}"],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    if proc.returncode != 0:
        # Shallow PR checkouts may have both endpoints but no merge base.
        proc = subprocess.run(
            ["git", "diff", "--name-only", base, head],
            cwd=REPO_ROOT,
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        )
    return [REPO_ROOT / line for line in proc.stdout.splitlines() if line]


def _is_documentation_path(rel: Path) -> bool:
    if rel.suffix.lower() in {".md", ".rst"}:
        return True
    return rel.parts[:1] == ("docs",)


def selected_pytest_targets(paths: list[Path]) -> list[str]:
    if not paths:
        return [ALL_TESTS_TARGET]

    graph = internal_import_graph()
    changed_modules: set[str] = set()
    direct_test_targets: set[str] = set()
    saw_documentation = False

    for path in paths:
        try:
            rel = path.resolve().relative_to(REPO_ROOT)
        except ValueError:
            return [ALL_TESTS_TARGET]

        if _is_documentation_path(rel):
            saw_documentation = True
            continue

        top = rel.parts[0]
        if top in {".github", "scripts"}:
            return [ALL_TESTS_TARGET]
        if top == "environment" or rel.name in {
            "pyproject.toml",
            "run_tests.sh",
        }:
            return [ALL_TESTS_TARGET]
        if top == "tests":
            if rel.name in {"conftest.py", "claims.py"}:
                return [ALL_TESTS_TARGET]
            if rel.suffix == ".py" and rel.name.startswith("test_"):
                direct_test_targets.add(str(rel))
            continue
        if top == "cosmic_foundry":
            module = module_for_path(path)
            if module is None:
                return [ALL_TESTS_TARGET]
            changed_modules.add(module)
            continue

    impacted = reverse_reachable(graph, changed_modules) if changed_modules else set()
    graph_test_targets = {
        str(path_for_module(module).relative_to(REPO_ROOT))
        for module in impacted
        if module.startswith("tests.test_") and path_for_module(module) is not None
    }
    targets = direct_test_targets | graph_test_targets
    if changed_modules and not graph_test_targets:
        return [ALL_TESTS_TARGET]
    if saw_documentation and not targets:
        return [DOC_ONLY_TEST_TARGET]
    return sorted(targets) if targets else [ALL_TESTS_TARGET]


def main() -> int:
    base = sys.argv[1] if len(sys.argv) > 1 else "origin/main"
    print(" ".join(selected_pytest_targets(changed_files(base))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
