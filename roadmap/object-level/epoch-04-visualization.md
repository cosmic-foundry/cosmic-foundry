# Epoch 4 — Visualization and science communication

> Part of the [Cosmic Foundry roadmap](../index.md).

The visualization stack that every physics epoch will feed into —
and the public-facing science-communication surface. Promoted out
of the former Stretch Epoch 12 (subgrid physics and observables,
now Stretch Epoch 13) because excellence at visualization is a core
engine requirement, and because rendering-layer and output-format
choices must precede the first physics module to avoid expensive
rewrites. Builds directly on the Zarr writer delivered in Epoch 2
and the house-style scaffolding from Epoch 0.

- Unit-aware plotting layer over `unyt`, bridging `astropy.units`
  at the engine boundary.
- In-engine rendering primitives in JAX — camera, 2-D slice
  sampler, 3-D volume raymarcher, particle projector — shared
  between batch CPU / GPU renders and the browser viewer so the
  same kernels drive both.
- WebGPU viewer package (with WebGL2 fallback) that consumes
  Zarr tile pyramids and glTF geometry. Shaders authored once in
  WGSL and transpiled for WebGL2.
- MyST-NB explainer template embedding the viewer as an
  interactive widget; `sphinx-design` gallery page published as
  part of the docs site.
- Visual-regression harness upgraded from the Epoch 0 stub:
  `pytest-mpl` for figures, an SSIM-based diff harness for
  renders and short movies, references in Git LFS.
- Accessibility and performance budgets from ADR-0006
  (WCAG 2.2 AA, colorblind-safe palettes, mobile LCP < 2.5 s,
  per-dataset bytes-on-wire) codified as CI checks against the
  public gallery build.
- Public gallery (GitHub Pages) seeded with the Epoch 2 AMR
  advection test as its first interactive entry; later physics
  epochs each add one canonical live demo.

**Exit criterion:** the Epoch 2 AMR advection test renders as a
live WebGPU page on the public docs site from checkpoint data
with no hand editing; the page meets the accessibility and
performance budgets under CI; and the visual-regression harness
is green on figure, render, and movie references.
