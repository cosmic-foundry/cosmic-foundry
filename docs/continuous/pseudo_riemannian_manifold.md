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

# `PseudoRiemannianManifold`

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
    return Markdown("```python\n" + inspect.getsource(obj).strip() + "\n```")

def _mro_chain(cls):
    chain = [c for c in reversed(inspect.getmro(cls)) if c is not object]
    return Markdown(" → ".join(f"`{c.__qualname__}`" for c in chain))
```

```{code-cell} python
:tags: [remove-input]
from cosmic_foundry.continuous.pseudo_riemannian_manifold import PseudoRiemannianManifold
import test_pseudo_riemannian_manifold as _mod

display(_mro_chain(PseudoRiemannianManifold))
display(_doc(PseudoRiemannianManifold))
```

## Implementations

```{code-cell} python
:tags: [remove-input]
_impls = sorted(
    [(n, c) for n, c in inspect.getmembers(_mod, inspect.isclass)
     if not n.startswith("_") and c.__module__ == _mod.__name__],
    key=lambda x: inspect.getsourcelines(x[1])[1],
)
for _, cls in _impls:
    display(_src(cls))
```

## Tests

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
```
