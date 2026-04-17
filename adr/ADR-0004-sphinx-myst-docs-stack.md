# ADR-0004 — Sphinx + MyST-NB documentation stack

- **Status:** Proposed
- **Date:** 2026-04-14

> **Note on status.** Of the five Epoch-0 seed ADRs, the
> documentation stack is the one expected to iterate the most as
> real content lands. The decision below is Proposed, not Accepted:
> the Sphinx + MyST-NB foundation is firm, but theme, styling, and
> the interactive-embed path are open to revision in follow-up ADRs
> without disrupting prose, notebooks, or the API reference.

## Context

Cosmic Foundry's roadmap ([`roadmap/index.md`](../roadmap/index.md) §4,
["Documentation"](../roadmap/epoch-00-bootstrap.md) §0.5) commits to
documentation that advances alongside code rather than lagging behind
it: a user guide, theory manual, and API reference built from the
same repository. ADR-0006 (visualization stack) additionally relies
on the docs engine to host MyST notebooks with interactive
explainers, the public gallery, and the WebGPU viewer embeds.

Two properties the docs stack must deliver:

- **Executable notebooks alongside prose.** The educational surface
  is a core engine requirement (ADR-0006); reference docs need to
  run worked examples, render figures under the house colormap
  (ADR-0006), and produce plotfile / Zarr outputs that the web
  viewer consumes. A static-page-only docs engine cannot deliver
  the explainer experience the project commits to.
- **Python-native authoring.** ADR-0001 commits to Python as the
  single source language; docs authoring should stay in the same
  ecosystem rather than introducing a separate build toolchain
  (e.g. a Jekyll or Hugo site alongside Sphinx).

Against those constraints, the candidate stacks are Sphinx (with
MyST and MyST-NB for Markdown + executable notebooks), MkDocs
(with `mkdocs-material` and `mkdocs-jupyter`), and Jupyter Book
(which is now itself a thin layer over Sphinx + MyST-NB). Sphinx
has the broadest scientific-Python ecosystem (`sphinx-design`,
`sphinx-autodoc2`, `numpydoc`, theory-manual extensions, full
cross-reference machinery) and the mature autodoc / API-reference
path that MkDocs still handles only through third-party plugins.
Jupyter Book reduces to Sphinx + MyST-NB with additional
configuration, so choosing Sphinx directly keeps the stack one
layer shallower without losing capability.

## Decision

Documentation is authored and built with **Sphinx + MyST-NB**,
extended with **`sphinx-design`** for layout primitives (cards,
grids, tabbed content for the public gallery and explainer pages).
`myst-parser` handles Markdown prose; `myst-nb` executes notebooks
in-build and renders their outputs. The rendering theme is
intentionally left open (see the "Rendered-page aesthetic" bullet
below) and lands with the first substantial docs PR.

- **Author format.** MyST-flavoured Markdown for prose pages;
  Jupyter notebooks (`.ipynb`) for executable content. Reference
  pages may mix both via MyST-NB directives.
- **API reference.** `sphinx-autodoc2` (not classic `autodoc`) is
  the autodoc extension of record for Cosmic Foundry's typed Python
  API, because it handles modern type annotations without
  importing the package at docs-build time.
- **Docstring style.** NumPy format, enforced through `numpydoc`.
- **Build command.** `sphinx-build -W docs docs/_build/html` —
  warnings are errors. This is a required step in the Epoch-0
  exit criteria and in CI once the docs tree lands.
- **Publishing.** Local `sphinx-build` only in Epoch 0. Read the
  Docs / GitHub Pages publishing is deferred until there is real
  content to host; the decision to adopt one or the other is left
  to a future ADR so that neither the hosting surface nor the URL
  shape is frozen prematurely.
- **Interactivity model.** Interactive content is **parameter-
  driven**, not code-driven: the reader manipulates sliders,
  dropdowns, or similar controls, and live simulation outputs are
  computed in the browser by **engine-authored** WebGPU / WASM
  artifacts shipped with the docs build — not by any browser-side
  interpreter for reader-written code. The distinction is who
  authors the code, not when it runs: engine-authored WGSL compute
  shaders (transpiled once from JAX kernels per ADR-0006) and
  pre-compiled WASM modules execute live against whatever values
  the widgets carry, producing genuinely live simulations; a
  browser-side Python interpreter (`pyodide`, `JupyterLite`) that
  would let the reader author and run arbitrary code is **out of
  scope**. The WebGPU viewer (ADR-0006) is the default surface for
  interactive simulation renderings; `holoviews` + `bokeh` embeds
  are the notebook-internal fallback for 1-D / 2-D parameter
  studies that do not need the viewer.
- **Rendered-page aesthetic.** Rendered pages must read as modern
  documentation, not as Jupyter notebooks. MyST-NB is configured
  to hide input-cell prompts (`In [ ]:`) and cell numbering; by
  default, pages surface prose and figure / widget output, with
  source cells collapsed behind an opt-in "show code" affordance
  for readers who want to inspect the derivation. The specific
  theme and CSS polish are intentionally not pinned by this ADR —
  `furo` and `pydata-sphinx-theme` are the two finalists, with the
  choice made alongside the first substantial docs PR.
- **Cross-referencing.** `intersphinx` is enabled against NumPy,
  SciPy, JAX, h5py, and `unyt` from day one so API-reference cross-
  links into upstream docs are cheap.

The docs tree itself is seeded in Epoch 0 per §0.5 with
`index.md`, `getting-started.md`, `contributing.md`,
`coding-standards.md`, `theory/` (empty per-module stubs), `api/`
(autodoc entry point), and `adr/index.md` generated from the ADR
directory.

## Consequences

- **Positive.** One build system for prose, executable notebooks,
  and API reference; the same authoring flow serves the educational
  surface, the internal reference manual, and ADR indexing. MyST-NB
  executes examples at build time, which converts "docs drift" from
  a human-spotting problem into a CI failure. Cross-reference
  ecosystem (`intersphinx`, `sphinx-design`, `sphinx-autodoc2`) is
  the largest in scientific Python. Both theme finalists (`furo`,
  `pydata-sphinx-theme`) ship with accessibility defaults that
  reduce the ADR-0006 accessibility-budget work; confirming the
  chosen theme still clears WCAG 2.2 AA is a check at docs-PR time.
- **Negative.** Sphinx's build is heavier than MkDocs'; a full
  notebook-execution pass through the docs tree is minutes rather
  than seconds once physics notebooks land. Mitigated by MyST-NB's
  cache (`jupyter-cache`, already pinned in `environment/cosmic_
  foundry.yml`) and by running docs on a dedicated CI job rather
  than inside the pre-commit hook. MyST syntax is close to but not
  identical to CommonMark, so Markdown written against GitHub's
  renderer occasionally needs small adjustments for the docs build.
- **Neutral.** The docs tree lives in `docs/` with Sphinx sources;
  markdown files at the repository root (README.md, AI.md,
  ROADMAP.md, RESEARCH.md, this ADR) remain GitHub-rendered and
  are intentionally kept out of the Sphinx build — they are
  navigation-layer documents, not reference material. Linking
  between the two surfaces is one-way: the docs tree can reference
  repo-root files by relative path; the repo-root files do not
  reach into `docs/_build/`.
- **Neutral — open follow-ups.** The ADR is Proposed; three
  sub-decisions land alongside the first substantial docs PR:
  exact theme (`furo` vs `pydata-sphinx-theme` vs a heavier
  custom skin), the CSS treatment that delivers the "not
  notebook-looking" aesthetic, and the widget layout conventions
  that keep parameter-driven simulation pages consistent across
  physics modules.

## Alternatives considered

- **MkDocs + `mkdocs-material` + `mkdocs-jupyter`.** Faster build,
  simpler theme story, but the autodoc / API-reference path is
  a third-party plugin (`mkdocstrings`) rather than first-class,
  and the ecosystem for theory-manual extensions (math rendering,
  citation machinery, intersphinx) is thinner. Rejected because
  the API reference and theory manual are structural load-bearers.
- **Jupyter Book.** Reduces to Sphinx + MyST-NB + a default
  configuration. Rejected as adding one layer on top of the stack
  we already plan to use, without a capability the stack beneath
  does not provide.
- **Plain Markdown + GitHub rendering.** Simplest option but does
  not execute notebooks, lacks cross-referencing, and cannot host
  the WebGPU viewer embeds ADR-0006 depends on. Rejected as
  insufficient for the educational surface.
- **ReStructuredText instead of MyST.** RST has Sphinx-native
  tooling and the widest ecosystem, but is less approachable for
  external contributors and prevents authors from writing in the
  same Markdown flavor used elsewhere in the repository.
  Rejected because the contributor ramp-up cost compounds across
  every docs PR.
- **Browser-side Python (`pyodide` / `JupyterLite`).** Enables
  narratives in which the reader edits and runs arbitrary Python.
  Rejected — but note the thing rejected is *reader-authored*
  code, not live computation per se: the WebGPU viewer path
  (ADR-0006) still runs real simulations live in the browser, just
  against engine-authored WGSL / WASM. A full Python runtime would
  add 10–50 MB of assets per page, bring sandboxing and arbitrary-
  code-execution responsibilities the project does not want to
  take on, and buy an interactivity mode (reader-edits-code) that
  the educational surface does not require.
- **Classic Jupyter-notebook rendering.** Default MyST-NB output
  surfaces `In [ ]:` prompts, cell numbering, and notebook
  chrome. Rejected because the educational surface targets
  readers who do not want to parse notebook semantics to follow a
  physics explainer. Rendered pages should look like modern
  documentation; notebook-cell chrome is explicitly hidden by
  default.

## Amendments

- **2026-04-16 — Hosting and multi-repo URL strategy.** The original
  decision deferred "Read the Docs / GitHub Pages publishing … until
  there is real content to host." The docs tree has now been seeded
  (Epoch 0) and CI validates it on every PR; the deferred decision
  is resolved here.

  **Hosting platform:** GitHub Pages, deployed via GitHub Actions
  using `actions/upload-pages-artifact` + `actions/deploy-pages`
  with the built-in `GITHUB_TOKEN` permissions — no PAT or deploy
  key required.

  **Trigger:** Push to `main` only. PRs continue to build and
  linkcheck but never deploy. There is one "latest" build tracking
  `main`; versioned docs are deferred until the project cuts a
  stable release (consistent with the "no stable APIs yet" policy in
  AI.md).

  **URL structure:** GitHub Pages serves `cosmic-foundry.github.io/`
  only from a repo named `cosmic-foundry.github.io`; all other repos
  are served under their repo name as a subpath. Engine docs
  therefore live at:

  > `https://cosmic-foundry.github.io/cosmic-foundry/`

  **Multi-repo strategy:** each sibling repo (`cosmic-observables`,
  `stellar-foundry`, and any future additions) deploys its own docs
  independently to `cosmic-foundry.github.io/<repo-name>/` via the
  same workflow pattern. Repos cross-reference each other via
  `intersphinx`; entries are added to each repo's `conf.py` once
  the target has a published build. No central portal repo
  (`cosmic-foundry.github.io`) is created at this time — the
  overhead is not warranted while the sibling repos are early-stage.
  The portal is a natural follow-up once all three repos have
  substantial published docs.

  **`html_baseurl`:** set in `conf.py` so Sphinx generates correct
  canonical URLs and sitemap entries for the `/cosmic-foundry/`
  subpath.

## Cross-references

- [`roadmap/index.md`](../roadmap/index.md) §2, §4 (Documentation).
- [`roadmap/epoch-00-bootstrap.md`](../roadmap/epoch-00-bootstrap.md)
  §0.5 (Documentation scaffolding).
- ADR-0001 (Python-only engine) — constrains the authoring stack
  to stay Python-native.
- ADR-0006 (Visualization and science-communication stack) —
  depends on this docs engine to host interactive explainers, the
  public gallery, and the WebGPU viewer embeds.
