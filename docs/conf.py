"""Sphinx configuration for Cosmic Foundry documentation."""

import sys
import time
from pathlib import Path

# Allow importing the package without a full install when building docs locally.
sys.path.insert(0, str(Path(__file__).parent.parent))
# Allow importing local Sphinx extensions from docs/_ext.
sys.path.insert(0, str(Path(__file__).parent))
# Allow notebooks to import from the tests directory.
sys.path.insert(0, str(Path(__file__).parent.parent / "tests"))

from cosmic_foundry._version import __version__  # noqa: E402
from scripts.gen_capability_atlas_docs import (
    generate as _gen_capability_atlas_docs,
)  # noqa: E402
from scripts.gen_continuous_docs import generate as _gen_continuous_docs  # noqa: E402
from scripts.gen_validation_docs import generate as _gen_validation_docs  # noqa: E402


def _time_docs_step(label, func):
    started = time.perf_counter()
    try:
        return func()
    finally:
        elapsed = time.perf_counter() - started
        print(f"[docs-timing] config:{label} {elapsed:.3f}s", file=sys.stderr)


_time_docs_step("gen_capability_atlas_docs", _gen_capability_atlas_docs)
_time_docs_step("gen_continuous_docs", _gen_continuous_docs)
_time_docs_step("gen_validation_docs", _gen_validation_docs)

# ---------------------------------------------------------------------------
# Project information
# ---------------------------------------------------------------------------

project = "Cosmic Foundry"
author = "Cosmic Foundry contributors"
release = __version__
version = __version__

# ---------------------------------------------------------------------------
# General configuration
# ---------------------------------------------------------------------------

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.duration",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_design",
    "sphinx_copybutton",
    "_ext.build_timing",
    # sphinx-autodoc2 wired up once there is substantial API surface to expose
]

exclude_patterns = ["_build", "**.ipynb_checkpoints"]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
]

nb_execution_mode = "force"  # always re-execute; CI fails if any notebook cell raises

duration_n_slowest = 0  # show all document read durations

# ---------------------------------------------------------------------------
# HTML output
# ---------------------------------------------------------------------------

html_theme = "furo"
html_title = f"Cosmic Foundry {version}"

# Canonical base URL — must match the GitHub Pages subpath.
html_baseurl = "https://cosmic-foundry.github.io/cosmic-foundry/"

# ---------------------------------------------------------------------------
# Intersphinx
# ---------------------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "jax": ("https://jax.readthedocs.io/en/latest", None),
}
