"""Generate MyST-NB pages from Jupytext percent-format scripts in validation/.

For each `validation/{scenario}/` directory containing a `*.py` script
(not prefixed `test_` or `__`), this script converts the percent-format
cells to MyST-NB code cells and writes the result to `docs/{scenario}/`.

Cell markers of the form `# %% Name` cause the docstring of `Name` (looked
up from the script's imports) to be inserted as prose before that cell.
The module docstring's first line becomes the page title; remaining lines
become the page introduction.

Generated files are build artifacts: gitignored, not committed.
Run automatically from docs/conf.py before Sphinx processes sources.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).parent.parent
_VALIDATION = _PROJECT_ROOT / "validation"
_DOCS_OUT = _PROJECT_ROOT / "docs"

# Ensure validation/ is importable when running the generator standalone.
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_FRONTMATTER = """\
---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---
"""


def _extract_docstring(source: str) -> tuple[str, str] | None:
    """Return (title, intro) from the module docstring, or None."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    doc = ast.get_docstring(tree)
    if not doc:
        return None
    lines = doc.splitlines()
    title = lines[0].rstrip(".")
    intro = "\n".join(lines[2:]).strip() if len(lines) > 2 else ""
    return title, intro


def _build_name_map(source: str) -> dict[str, Any]:
    """Map names imported in *source* to their live objects."""
    tree = ast.parse(source)
    name_map: dict[str, Any] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            try:
                mod = importlib.import_module(node.module)
            except ImportError:
                continue
            for alias in node.names:
                obj = getattr(mod, alias.name, None)
                if obj is not None:
                    name_map[alias.asname or alias.name] = obj
    return name_map


def _split_cells(source: str) -> list[tuple[str | None, str]]:
    """Split *source* on `# %%` markers; return (tag, body) pairs."""
    chunks = source.split("\n# %%")
    cells = []
    for chunk in chunks:
        stripped = chunk.strip()
        if not stripped or stripped.startswith(('"""', "'''")):
            continue
        first_newline = chunk.find("\n")
        if first_newline == -1:
            continue
        tag = chunk[:first_newline].strip() or None
        body = chunk[first_newline:].strip()
        if body:
            cells.append((tag, body))
    return cells


def _doc_body(obj: Any) -> str | None:
    """Return the docstring after the one-liner, for use as doc page prose."""
    doc = inspect.getdoc(obj)
    if not doc:
        return None
    parts = doc.split("\n\n", 1)
    return parts[1].strip() if len(parts) > 1 else None


def _render_page(
    title: str,
    intro: str,
    cells: list[tuple[str | None, str]],
    name_map: dict[str, Any],
) -> str:
    parts = [f"{_FRONTMATTER.strip()}\n\n# {title}"]
    if intro:
        parts.append(intro)
    parts.append("---")
    for tag, body in cells:
        if tag:
            obj = name_map.get(tag)
            doc = _doc_body(obj) if obj is not None else None
            if doc:
                parts.append(doc)
        parts.append(f"```{{code-cell}} python\n{body}\n```")
    return "\n\n".join(parts) + "\n"


def _render_index(scenario: str, stems: list[str]) -> str:
    title = scenario.replace("_", " ").title()
    entries = "\n".join(stems)
    return f"# {title}\n\n```{{toctree}}\n:maxdepth: 1\n\n{entries}\n```\n"


def generate(out_root: Path = _DOCS_OUT) -> None:
    """Write per-scenario pages and index files into *out_root*."""
    for scenario_dir in sorted(_VALIDATION.iterdir()):
        if not scenario_dir.is_dir() or scenario_dir.name.startswith("_"):
            continue

        stems = []
        for script in sorted(scenario_dir.glob("*.py")):
            if script.stem.startswith("test_") or script.stem.startswith("_"):
                continue
            source = script.read_text()
            result = _extract_docstring(source)
            if result is None:
                continue
            title, intro = result
            name_map = _build_name_map(source)
            cells = _split_cells(source)
            if not cells:
                continue

            out_dir = out_root / scenario_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)
            page = _render_page(title, intro, cells, name_map)
            (out_dir / f"{script.stem}.md").write_text(page)
            stems.append(script.stem)

        if stems:
            out_dir = out_root / scenario_dir.name
            (out_dir / "index.md").write_text(_render_index(scenario_dir.name, stems))


if __name__ == "__main__":
    generate()
