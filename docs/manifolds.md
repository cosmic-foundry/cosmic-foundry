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

# Manifolds

```{code-cell} python
:tags: [remove-input]
import importlib.util
import sys
from pathlib import Path

# cosmic_foundry is installed in editable mode; its location anchors the project root.
_root = Path(importlib.util.find_spec("cosmic_foundry").origin).parent.parent
if str(_root / "tests") not in sys.path:
    sys.path.insert(0, str(_root / "tests"))

import inspect
from IPython.display import Markdown

def _doc(obj):
    return Markdown(inspect.getdoc(obj))

def _src(obj):
    return Markdown("```python\n" + inspect.getsource(obj).strip() + "\n```")
```

```{code-cell} python
:tags: [remove-input]
from cosmic_foundry.continuous.manifold import Manifold
_doc(Manifold)
```

```{code-cell} python
from test_manifold_hierarchy import assert_manifold_is_abstract
_src(assert_manifold_is_abstract)
```

```{code-cell} python
assert_manifold_is_abstract()
```

## `PseudoRiemannianManifold`

```{code-cell} python
:tags: [remove-input]
from cosmic_foundry.continuous.pseudo_riemannian_manifold import PseudoRiemannianManifold
_doc(PseudoRiemannianManifold)
```

```{code-cell} python
from test_manifold_hierarchy import assert_pseudo_riemannian_manifold_is_abstract
_src(assert_pseudo_riemannian_manifold_is_abstract)
```

```{code-cell} python
assert_pseudo_riemannian_manifold_is_abstract()
```

## Concrete implementations

```{code-cell} python
from test_manifold_hierarchy import FlatR3, MinkowskiR4
_src(FlatR3)
```

```{code-cell} python
_src(MinkowskiR4)
```

```{code-cell} python
from test_manifold_hierarchy import (
    assert_flat_r3_isinstance_chain,
    assert_flat_r3_ndim_derived_from_signature,
    assert_minkowski_r4_isinstance_chain,
    assert_minkowski_r4_ndim_derived_from_signature,
)

_src(assert_flat_r3_isinstance_chain)
```

```{code-cell} python
assert_flat_r3_isinstance_chain()
assert_flat_r3_ndim_derived_from_signature()
assert_minkowski_r4_isinstance_chain()
assert_minkowski_r4_ndim_derived_from_signature()
```

## Disjointness from the discrete hierarchy

```{code-cell} python
from test_manifold_hierarchy import assert_manifold_branch_disjoint_from_indexed_set_branch
_src(assert_manifold_branch_disjoint_from_indexed_set_branch)
```

```{code-cell} python
assert_manifold_branch_disjoint_from_indexed_set_branch()
```
