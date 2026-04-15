"""Sphinx configuration for Cosmic Foundry documentation."""

import sys
from pathlib import Path

# Allow importing the package without a full install when building docs locally.
sys.path.insert(0, str(Path(__file__).parent.parent))

from cosmic_foundry._version import __version__  # noqa: E402

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
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_design",
    "sphinx_copybutton",
    # sphinx-autodoc2 wired up once there is substantial API surface to expose
]

exclude_patterns = ["_build", "**.ipynb_checkpoints"]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
]

nb_execution_mode = "off"  # notebooks executed explicitly, not on every build

# ---------------------------------------------------------------------------
# HTML output
# ---------------------------------------------------------------------------

html_theme = "furo"
html_static_path = ["_static"]
html_title = f"Cosmic Foundry {version}"

# ---------------------------------------------------------------------------
# Intersphinx
# ---------------------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "jax": ("https://jax.readthedocs.io/en/latest", None),
}

# ---------------------------------------------------------------------------
# Link checking (sphinx-build -b linkcheck)
# ---------------------------------------------------------------------------
#
# Own-repo URLs pointing at the main branch are validated against the working
# tree by scripts/ci/check_markdown_links.py, not over HTTP. Linkcheck would
# otherwise 404 on any link added in the same PR as its target file (main
# does not yet contain the file at the time the PR runs). Keep this list
# for true external URLs only.
linkcheck_ignore = [
    r"^https://github\.com/cosmic-foundry/cosmic-foundry/(?:blob|tree|raw)/main/",
]
