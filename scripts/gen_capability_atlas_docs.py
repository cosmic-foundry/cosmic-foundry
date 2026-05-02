"""Generate capability-atlas documentation from structural claims."""

from __future__ import annotations

from pathlib import Path

from tests.test_structure import (
    _render_capability_atlas,
    _render_capability_atlas_plots,
)

_PROJECT_ROOT = Path(__file__).parent.parent
_DOCS_OUT = _PROJECT_ROOT / "docs"
_PLOTS_OUT = _DOCS_OUT / "capability_atlas_plots"


def generate(out_root: Path = _DOCS_OUT) -> None:
    """Write the capability atlas page and generated plot assets."""
    plots_out = out_root / "capability_atlas_plots"
    plots_out.mkdir(parents=True, exist_ok=True)
    for stale_plot in plots_out.glob("*.svg"):
        stale_plot.unlink()

    (out_root / "capability_atlas.md").write_text(_render_capability_atlas())
    for name, svg in _render_capability_atlas_plots().items():
        (plots_out / name).write_text(svg)


if __name__ == "__main__":
    generate()
