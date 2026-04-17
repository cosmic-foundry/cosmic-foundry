"""Visual regression test: sinusoid rendered with the house colormap.

This is the seed test for the pytest-mpl harness. Later physics modules add
cards to the docs gallery and corresponding tests here.

Run with image comparison:
    pytest --mpl tests/visual/

Generate/update baselines:
    pytest --mpl-generate-path=tests/visual/baseline tests/visual/
"""

from __future__ import annotations

import numpy as np
import pytest

# House colormap: viridis is colorblind-safe and ships with matplotlib.
# Swap to cmasher / cmocean once those dependencies are added to the palette.
HOUSE_CMAP = "viridis"


@pytest.mark.visual
@pytest.mark.mpl_image_compare(
    baseline_dir="baseline",
    filename="sinusoid_viridis.png",
    tolerance=2,
    style="default",
)
def test_sinusoid_house_colormap() -> object:  # returns Figure
    """Sinusoid colored by phase under the house colormap."""
    import matplotlib.pyplot as plt

    x = np.linspace(0, 2 * np.pi, 256)
    y = np.sin(x)

    fig, ax = plt.subplots(figsize=(6, 3))
    sc = ax.scatter(x, y, c=x, cmap=HOUSE_CMAP, s=8, linewidths=0)
    fig.colorbar(sc, ax=ax, label="phase (rad)")
    ax.set_xlabel("x")
    ax.set_ylabel("sin(x)")
    ax.set_title("House colormap smoke test")
    fig.tight_layout()
    return fig
