# Accessibility and performance targets

This page records the targets the public gallery will enforce once it carries
real simulation outputs. The targets are set now so later epochs cannot drift
from them silently.

## Accessibility

- **Contrast:** WCAG 2.2 AA (minimum 4.5:1 for normal text, 3:1 for large
  text and UI components).
- **Colour palettes:** all figure colormaps must be colourblind-safe.
  Perceptual maps from `cmasher` and `cmocean` are the preferred sources;
  matplotlib's `viridis`, `plasma`, `inferno`, and `cividis` are acceptable
  fallbacks.
- **Alt text:** every figure in the docs must carry a descriptive `alt`
  attribute.

## Performance budget (web gallery)

- **LCP:** under 2.5 s on a 4G network profile (mobile).
- **Bytes on wire:** tiled datasets served via Zarr v3; per-tile budget TBD
  once the first dataset is wired up (Epoch 3+).
- **WebGPU / WebGL2:** volume renders target 60 fps on a mid-range GPU;
  WebGL2 fallback required.
