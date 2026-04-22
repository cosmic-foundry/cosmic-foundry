"""Generate MyST-NB pages from Jupytext percent-format scripts in validation/.

For each `validation/{scenario}/` directory containing a `*.py` script
(not prefixed `test_` or `__`), this script converts the percent-format
cells to MyST-NB code cells and writes the result to `docs/{scenario}/`.

The module docstring of each script becomes the page title; `# %%` markers
become code-cell boundaries. Generated files are build artifacts: gitignored,
not committed.

Run automatically from docs/conf.py before Sphinx processes sources.
"""

from __future__ import annotations

import ast
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
_VALIDATION = _PROJECT_ROOT / "validation"
_DOCS_OUT = _PROJECT_ROOT / "docs"

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


def _extract_docstring(source: str) -> str | None:
    """Return the module-level docstring from *source*, or None."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    return ast.get_docstring(tree)


def _split_cells(source: str) -> list[str]:
    """Split *source* on `# %%` markers; return non-empty cell bodies."""
    raw = source.split("\n# %%")
    cells = []
    for chunk in raw:
        # Strip the leading newline left by the split and any trailing whitespace.
        body = chunk.strip()
        # Skip the first chunk if it is only the module docstring (it becomes title).
        if body.startswith('"""') or body.startswith("'''"):
            continue
        if body:
            cells.append(body)
    return cells


def _render_page(title: str, cells: list[str]) -> str:
    code_cells = "\n\n".join(f"```{{code-cell}} python\n{cell}\n```" for cell in cells)
    return f"{_FRONTMATTER.strip()}\n\n# {title}\n\n{code_cells}\n"


def _render_index(scenario: str, stems: list[str]) -> str:
    entries = "\n".join(stems)
    title = scenario.replace("_", " ").title()
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
            docstring = _extract_docstring(source)
            if docstring is None:
                continue
            title = docstring.splitlines()[0].rstrip(".")
            cells = _split_cells(source)
            if not cells:
                continue

            out_dir = out_root / scenario_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"{script.stem}.md").write_text(_render_page(title, cells))
            stems.append(script.stem)

        if stems:
            out_dir = out_root / scenario_dir.name
            (out_dir / "index.md").write_text(_render_index(scenario_dir.name, stems))


if __name__ == "__main__":
    generate()
