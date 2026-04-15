# ADR-0006 — Visualization and science-communication stack

- **Status:** Proposed
- **Date:** 2026-04-13

## Context

Cosmic Foundry treats excellence at visualization as a core
requirement: the engine's outputs must serve both high scientific
rigor and high educational impact, including through public-facing
web content. RESEARCH §6.10 originally cataloged visualization
only as "integrated analysis hooks (yt, VisIt, ParaView, Ascent)" —
a batch, desktop-era stance. The §6.11 research survey added
alongside this ADR (now at `research/06-11-visualization.md` after
the docs split) reviews the modern landscape (Python plotting,
browser-native renderers, unit-aware plotting, tile and streaming
formats, perceptual colormaps, visual-regression tooling, and
science-communication surfaces).

Because early epochs commit the engine to HDF5 + plotfile outputs,
the viz stack's output-format decisions must be made *before* those
writers are designed — otherwise the engine ships web-hostile
outputs or duplicates its I/O paths.

ADR-0001..0005 (the five Epoch-0 seed ADRs reserved in
`roadmap/epoch-00-bootstrap.md` §0.6) codify commitments this ADR
builds on. This ADR uses number 0006 to preserve that reservation.

## Decision

The visualization and science-communication stack for Cosmic Foundry
comprises the following initial choices, permissively licensed and
pure-Python or browser-native where possible:

- **Perceptual colormaps.** `cmasher` (CVD-safe, scientific
  defaults) is the house library; `cmocean` and matplotlib
  perceptually-uniform maps (`viridis`, `cividis`, `inferno`,
  `magma`) are the only fallbacks sanctioned for publication
  figures. No rainbow / `jet`-era maps.
- **Unit-aware plotting.** Physical quantities flow through `unyt`
  (BSD) so axes, colorbars, and interactive widgets always label
  units without manual bookkeeping. `astropy.units` is consumed at
  the engine boundary; internal representation is `unyt`.
- **Static and interactive Python plotting.** `matplotlib` for
  publication figures; `holoviews` + `bokeh` for interactive
  notebook and dashboard work; `datashader` for large particle and
  field renders that exceed direct-canvas limits.
- **Desktop 3-D rendering.** `pyvista` (MIT) as the default VTK
  wrapper; `vispy` (BSD) as a GPU-accelerated alternative for
  smoothly-animated particle and volume work. Integration with
  `yt` (BSD-3) is a bridge consumed via plotfile/Zarr exports, not
  a substitute for the engine's own renderers.
- **In-engine rendering primitives.** A minimal JAX-implementable
  camera, 2-D slice sampler, 3-D volume raymarcher, and particle
  projector live inside the engine so that renderings reproduce
  under the same kernel backends as physics and can be ported to
  WebGPU shaders without a second implementation.
- **Web-facing streaming format.** `Zarr` v3 (MIT) with an
  OME-Zarr-style multiscale-pyramid convention is the canonical
  tile-friendly output. HDF5 remains the checkpoint format; Zarr
  is written alongside for analysis and web consumption. ADIOS2
  is retained as a future option behind a later ADR, not the
  web-facing path.
- **Browser rendering target.** `WebGPU` is the primary target;
  `WebGL2` is an automatic fallback. Engine-side shaders are
  authored once in WGSL and transpiled for WebGL2 where needed.
  `three.js` (MIT) for scene-graph primitives; `deck.gl` (MIT)
  for large particle / point-cloud layers; `regl` (MIT) kept on
  the shortlist for hand-tuned passes.
- **Educational surface.** MyST-NB + Sphinx + `sphinx-design`
  host reference docs. Interactive explainers ship as standalone
  static-site builds embedding the WebGPU viewer, authored as
  MyST notebooks and published to GitHub Pages alongside the
  release gallery. `pyodide` / `JupyterLite` are adopted where
  read-execute-explain narratives need Python in the browser.
- **Visual-regression testing.** `pytest-mpl` (BSD) anchors figure
  tests; an in-repo perceptual-image-diff harness (built on
  `scikit-image` SSIM, BSD) anchors volume-render and movie
  regressions. Reference images and short movies live in Git LFS
  and are updated only by an explicit tag-and-review workflow.
- **Accessibility and performance budgets.** The public site
  enforces CVD-safe palettes only; alt text on every figure;
  WCAG 2.2 AA contrast; mobile Largest-Contentful-Paint under
  2.5 s on a 4G profile for any explainer page; per-dataset
  bytes-on-the-wire budgets.

## Consequences

**Positive.** Output-format decisions (Zarr alongside HDF5) are
pinned before the Epoch 2 plotfile writer is designed, avoiding
future I/O rewrites. The web viewer and docs gallery populate as
soon as the first physics tests run, so "slick" content exists
early rather than arriving in a stretch epoch. The JAX commitment
is leveraged: camera and volume renderers are the same kernels on
CPU, GPU, and in the browser.

**Negative.** Two output formats (HDF5, Zarr) means dual-writing
during checkpoints; mitigated by a thin adapter. A WebGPU-first
stance temporarily loses some users on older browsers — hence the
WebGL2 fallback. Perceptual-diff test infrastructure is new work
not needed by engines that ship only HDF5 regressions.

**Neutral.** Licensing is uniformly permissive (BSD / MIT / PSF /
Apache); no viewer-side dependency introduces copyleft
obligations. Cesium and Mapbox are *not* adopted — their asset /
tile licensing is incompatible with the engine's permissive
stance and no astrophysical use case requires them.

## Alternatives considered

- **yt-first, hand-off rendering.** Delegating all rendering to
  yt removes in-repo renderer work but ties interactive browser
  content to yt's widget path (`widgyts`), which is research-
  grade rather than product-grade. Rejected because the
  educational surface requires interactive renderers we control.
- **ADIOS2 as the only viz-shaped output format.** ADIOS2's BP5
  engine is HPC-excellent but its browser story is immature.
  Zarr v3 ships to the browser today. ADIOS2 remains available
  for HPC analysis but is not the web-facing path.
- **WebGL2-first browser target.** Postponing WebGPU would
  simplify shader authoring but forecloses compute-shader-heavy
  volume rendering. WebGPU shipped across Chromium-, Gecko-, and
  WebKit-based browsers during 2024–2025; its maturity crossed
  the adoption threshold.
- **Pure batch visualization (no web surface).** Simplest
  delivery but incompatible with the core requirement of high
  educational impact through public-facing web content.
