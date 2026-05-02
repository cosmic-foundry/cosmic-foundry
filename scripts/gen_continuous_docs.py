"""Generate one MyST-NB notebook per ABC in cosmic_foundry/theory/continuous/.

For each module `theory/continuous/{stem}.py` with a matching `tests/test_{stem}.py`,
this script discovers the primary ABC defined in the module and writes
`docs/continuous/{stem}.md`. It also writes `docs/continuous/index.md`.

Run automatically from docs/conf.py before Sphinx processes sources.
The generated files are gitignored — they are build artifacts, not sources.
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
_CONTINUOUS = _PROJECT_ROOT / "cosmic_foundry" / "theory" / "continuous"
_TESTS = _PROJECT_ROOT / "tests"
_DOCS_OUT = _PROJECT_ROOT / "docs" / "continuous"

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

_SETUP_CELL = """\
```{code-cell} python
:tags: [remove-input]
import importlib.util
import inspect
import sys
from pathlib import Path
from IPython.display import Markdown, display

_root = Path(importlib.util.find_spec("cosmic_foundry").origin).parent.parent
if str(_root / "tests") not in sys.path:
    sys.path.insert(0, str(_root / "tests"))

def _doc(obj):
    return Markdown(inspect.getdoc(obj))

def _src(obj):
    return Markdown("```python\\n" + inspect.getsource(obj).strip() + "\\n```")

def _mro_chain(cls):
    chain = [c for c in reversed(inspect.getmro(cls)) if c is not object]
    return Markdown(" → ".join(f"`{c.__qualname__}`" for c in chain))
```\
"""

_HEADER_CELL = """\
```{{code-cell}} python
:tags: [remove-input]
from {module_path} import {class_name}
import {test_module} as _mod

display(_mro_chain({class_name}))
display(_doc({class_name}))
```\
"""

_IMPLS_CELL = """\
```{code-cell} python
:tags: [remove-input]
_impls = sorted(
    [(n, c) for n, c in inspect.getmembers(_mod, inspect.isclass)
     if not n.startswith("_") and c.__module__ == _mod.__name__],
    key=lambda x: inspect.getsourcelines(x[1])[1],
)
for _, cls in _impls:
    display(_src(cls))
```\
"""

_ASSERTS_CELL = """\
```{code-cell} python
:tags: [remove-input]
_asserts = sorted(
    [(n, f) for n, f in inspect.getmembers(_mod, inspect.isfunction)
     if n.startswith("assert_")],
    key=lambda x: x[1].__code__.co_firstlineno,
)
for _, fn in _asserts:
    display(_src(fn))
    fn()
```\
"""


def _find_primary_abc(stem: str) -> str | None:
    """Return the name of the primary ABC defined in continuous/{stem}.py."""
    module_path = f"cosmic_foundry.theory.continuous.{stem}"
    try:
        mod = importlib.import_module(module_path)
    except ImportError:
        return None
    abcs = [
        name
        for name, obj in inspect.getmembers(mod, inspect.isclass)
        if obj.__module__ == module_path and getattr(obj, "__abstractmethods__", None)
    ]
    if not abcs:
        return None
    # Prefer the name that matches the stem in PascalCase; fall back to first.
    pascal = "".join(w.capitalize() for w in stem.split("_"))
    return pascal if pascal in abcs else abcs[0]


def _render_page(class_name: str, module_path: str, test_module: str) -> str:
    header = _HEADER_CELL.format(
        class_name=class_name,
        module_path=module_path,
        test_module=test_module,
    )
    return (
        "\n\n".join(
            [
                _FRONTMATTER.strip(),
                f"# `{class_name}`",
                _SETUP_CELL,
                header,
                _IMPLS_CELL,
                _ASSERTS_CELL,
            ]
        )
        + "\n"
    )


def _render_index(stems: list[str]) -> str:
    if not stems:
        return """\
# Continuous layer

No auto-generated continuous-layer notebooks are available.  The generator
only emits notebooks for modules that have matching per-module test files.
"""
    entries = "\n".join(stems)
    return f"""\
# Continuous layer

```{{toctree}}
:maxdepth: 1

{entries}
```
"""


def generate(out_dir: Path = _DOCS_OUT) -> None:
    """Write per-ABC notebooks and index into *out_dir*."""
    out_dir.mkdir(parents=True, exist_ok=True)

    stems = []
    for path in sorted(_CONTINUOUS.glob("*.py")):
        stem = path.stem
        if stem == "__init__":
            continue
        if not (_TESTS / f"test_{stem}.py").exists():
            continue
        class_name = _find_primary_abc(stem)
        if class_name is None:
            continue
        module_path = f"cosmic_foundry.theory.continuous.{stem}"
        test_module = f"test_{stem}"
        page = _render_page(class_name, module_path, test_module)
        (out_dir / f"{stem}.md").write_text(page)
        stems.append(stem)

    current = {f"{stem}.md" for stem in stems} | {"index.md"}
    for path in out_dir.glob("*.md"):
        if path.name not in current:
            path.unlink()

    (out_dir / "index.md").write_text(_render_index(stems))


if __name__ == "__main__":
    generate()
