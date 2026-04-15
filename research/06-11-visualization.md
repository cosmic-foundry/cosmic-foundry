# 6.11 Visualization and science communication

> Part of the [Cosmic Foundry research notes](index.md).
> This is the subsection of §6 broken out into its own file because
> it surveys a fast-moving landscape (browser renderers, streaming
> formats, science-communication surfaces) and is expected to grow
> faster than the code survey. §-numbers are preserved so cross-
> references remain stable.

Visualization in the surveyed codes is almost entirely batch,
post-hoc, and desktop-shaped — plotfiles and HDF5 go out, analysis
frameworks (yt, VisIt, ParaView, Ascent) come in. Cosmic Foundry
treats visualization as a first-class engine capability serving
both scientific publication and public-facing science
communication, so the relevant landscape is broader than the
traditional HPC analysis stack surveyed in §6.10.

**Python analysis ecosystem.**

- `yt` (BSD-3) — de facto analysis standard for AMR and particle
  astrophysics output; unit-aware; Jupyter widgets via `widgyts`;
  supports AMReX, Enzo, Flash, GADGET, Arepo, RAMSES outputs.
- `napari` (BSD-3) — multi-dimensional image viewer, originally
  bio-imaging; useful for slice stacks and labeled volumes.
- `pyvista` (MIT) — Pythonic VTK wrapper; fast path to 3-D
  desktop rendering of meshes and point clouds.
- `vispy` (BSD-3) — GPU-accelerated 2-D/3-D Python library with
  OpenGL backends; good for smoothly-animated particle work.
- `k3d-jupyter` / `ipyvolume` (BSD / MIT) — in-notebook WebGL
  viewers for small-to-mid volumes.
- `datashader` (BSD-3) — server-side aggregation for datasets too
  large to rasterize client-side; pairs with HoloViews.
- `holoviews` + `panel` + `bokeh` (BSD) — interactive notebook /
  dashboard layer rendering to the browser.
- `plotly` / `dash` (MIT) — interactive plotting and dashboard
  framework; WebGL traces for particle work.

**Browser-native rendering.**

- `three.js` (MIT) — canonical WebGL scene graph; basis for most
  interactive science-communication experiences.
- `regl` (MIT) — functional-reactive WebGL wrapper; thinner and
  more hackable than three.js for custom shaders.
- `deck.gl` (MIT) — large-scale point/line/polygon layering;
  natural fit for particle clouds and halo catalogs.
- `WebGPU` — emerging standard shipping in all major browsers,
  with compute shaders suitable for volume raymarching and even
  lightweight in-client re-simulation.
- `CesiumJS` (Apache-2 library, asset licenses vary) — spherical
  geometry and streaming tiles; relevant for CMB or
  celestial-sphere visualizations, but asset-licensing footprint
  requires an explicit ADR before adoption.
- `OpenSeadragon` / DeepZoom / Neuroglancer tile servers — tiled
  pyramids for very large 2-D and 3-D fields.

**Streaming and viz-shaped output formats.**

- `Zarr` (MIT) — chunked, cloud- and browser-friendly array store;
  Zarr v3 adds sharding, consolidated metadata, and explicit
  codec conventions.
- `OME-Zarr` — Zarr conventions with multiscale pyramids, heavily
  used in bio-imaging; directly applicable to cosmological and
  radiation-field tile pyramids.
- `glTF` (MIT) — portable 3-D scene format; the right choice for
  any geometry (meshes, instanced particles) shipped to a web
  viewer.
- `Parquet` (Apache-2) — columnar format for particle and halo
  catalogs; streams well to notebook and browser consumers.
- `ADIOS2` — HPC-grade, excellent I/O but weak browser story;
  complementary to Zarr, not a substitute.

**Color and typography.**

- `matplotlib` perceptually-uniform maps (`viridis`, `cividis`,
  `inferno`, `magma`) — the scientific baseline.
- `cmasher` (MIT) — extended family of perceptually-uniform,
  color-vision-deficient-safe colormaps aimed at astrophysics.
- `cmocean` (MIT) — perceptual maps tuned for oceanography,
  equally appropriate for astrophysical diverging / cyclic data.
- Cinematic renders typically call for bespoke palettes informed
  by perceptual uniformity rather than the default `jet`-era
  rainbows still common in legacy codes.

**Unit-aware plotting.**

- `astropy.units` (BSD-3) — the astronomy-standard unit system.
- `unyt` (BSD-3) — lightweight unit system originally split out
  of yt; low-overhead, jittable-friendly, and what the analysis
  code in the ecosystem already carries.
- `pint` (BSD-3) — general-science alternative; adopted mainly
  outside astronomy.

**In-situ and data-as-movies paradigms.**

- `ALPINE / Ascent` (BSD) — in-situ visualization infrastructure
  linked directly into simulation codes; writes Cinema databases
  alongside traditional plotfiles.
- `Cinema` — parameterized image-database framework (Ahrens et
  al.); persists pre-rendered viewpoints so exploratory analysis
  is a database query, not a re-render.
- `ParaView Catalyst` and `VisIt libsim` — historical in-situ
  APIs; relevant mainly as algorithmic references, since Cosmic
  Foundry does not link compiled libraries.

**Science-communication surfaces.**

- `MyST-NB` + `sphinx-design` — executable narrative documents
  in the engine's docs pipeline; the baseline for theory manual
  content.
- `Jupyter Book` — one level above MyST-NB for book-length
  explainers with cross-references and executable content.
- `Observable` / `Distill.pub` — interactive-article idioms
  (scrollytelling, live widgets, paired math-and-simulation
  panels) that set the bar for educational impact.
- `Streamlit` / `Gradio` — fast-to-author interactive dashboards;
  adopted only where notebook-embedded HoloViews or Panel do not
  suffice.
- `pyodide` / `JupyterLite` — Python in the browser, enabling
  read-execute-explain narratives without a server.

**Visual-regression and testing infrastructure.**

- `pytest-mpl` (BSD) — baseline-image comparison for matplotlib
  figures; the default tool for figure tests.
- `scikit-image` SSIM (BSD) — perceptual image similarity,
  suitable for volume-render and movie diffs where pixel-exact
  comparison is brittle.
- `reg-viz`-style tooling — browser-side visual diffing for the
  web gallery and explainer pages.

**Implication for Cosmic Foundry.** The engine should (i) commit
to Zarr alongside HDF5 at the first plotfile writer so that
browser consumers are first-class; (ii) own a minimal in-engine
camera / slice-sampler / volume-raymarcher implemented in JAX so
the same code serves CPU, GPU, and WebGPU rendering; (iii) fix a
house style in colormaps, units, and typography before the first
physics module lands; and (iv) treat the science-communication
site — docs, gallery, explainers — as a product with
accessibility and performance budgets, not a byproduct of the
test suite.
