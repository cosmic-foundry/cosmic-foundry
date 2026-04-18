"""Visual regression test for a rendered Laplacian slice."""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.pointwise_roofline import make_phi, run_laplacian

HOUSE_CMAP = "viridis"


@pytest.mark.visual
@pytest.mark.mpl_image_compare(
    baseline_dir="baseline",
    filename="laplacian_slice_viridis.png",
    tolerance=2,
    style="default",
)
def test_laplacian_slice_house_colormap() -> object:  # returns Figure
    """Central z-slice of the Laplacian computed via Op.execute."""
    import matplotlib.pyplot as plt

    laplacian = np.asarray(run_laplacian(make_phi(32)))
    central_slice = laplacian[:, :, laplacian.shape[2] // 2]

    fig, ax = plt.subplots(figsize=(4, 4))
    image = ax.imshow(central_slice, origin="lower", cmap=HOUSE_CMAP)
    fig.colorbar(image, ax=ax, label="laplacian phi")
    ax.set_xlabel("i")
    ax.set_ylabel("j")
    ax.set_title("Seven-point Laplacian slice")
    fig.tight_layout()
    return fig
