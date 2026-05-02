"""Sphinx configuration for Cosmic Foundry documentation."""

import sys
from pathlib import Path

# Allow importing the package without a full install when building docs locally.
sys.path.insert(0, str(Path(__file__).parent.parent))
# Allow notebooks to import from the tests directory.
sys.path.insert(0, str(Path(__file__).parent.parent / "tests"))

from cosmic_foundry._version import __version__  # noqa: E402
from scripts.gen_capability_atlas_docs import (
    generate as _gen_capability_atlas_docs,
)  # noqa: E402
from scripts.gen_continuous_docs import generate as _gen_continuous_docs  # noqa: E402
from scripts.gen_validation_docs import generate as _gen_validation_docs  # noqa: E402

_gen_capability_atlas_docs()
_gen_continuous_docs()
_gen_validation_docs()

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

nb_execution_mode = "force"  # always re-execute; CI fails if any notebook cell raises

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
    # github.com is unreliable from CI (intermittent 502s); own-repo links are
    # validated against the working tree by scripts/ci/check_markdown_links.py.
    r"^https://github\.com/",
    # Publisher sites that block bots with 403s
    r"^https://arc\.aiaa\.org/",
    r"^https://doi\.org/10\.1002/",  # Wiley
    r"^https://doi\.org/10\.1146/",  # Annual Reviews
    r"^https://doi\.org/10\.1145/",  # ACM
    r"^https://doi\.org/10\.1177/",  # SAGE
    r"^https://dl\.acm\.org/",
    r"^https://onlinelibrary\.wiley\.com/",
    r"^https://journals\.sagepub\.com/",
    # Sites with broken/mismatched SSL certificates
    r"^https://2sn\.org/",
    r"^http://plutocode\.ph\.unito\.it/",
    r"^https://cactuscode\.org/",
    # NRC site — intermittently unreachable
    r"^https://www\.nrc\.gov/",
    # OSTI — intermittently unreachable (connection timeouts in CI)
    r"^https://www\.osti\.gov/",
    # MPC GitLab — intermittently unreachable from CI runners (connect timeout)
    r"^https://gitlab\.mpcdf\.mpg\.de/",
    # Confirmed-dead links in research/ — to be cleaned up separately
    r"^https://doi\.org/10\.1016/S0898122100003175$",  # dead DOI
    r"^https://www\.asme\.org/codes-standards/find-codes-standards/v-v-20",
]

# doi.org 301/302 redirects to publisher landing pages are expected.
linkcheck_allowed_redirects = {
    r"^https://doi\.org/": r".*",
}
