"""Figures derived from the Schwarzschild spacetime."""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np

from validation.schwarzschild.constants import (
    GPS_SEMI_MAJOR_AXIS,
    ICD_GPS200_FRACTIONAL_OFFSET,
    WGS84_C,
    WGS84_MU,
    WGS84_OMEGA_E,
    WGS84_R_E,
)


def time_dilation_figure() -> plt.Figure:
    """Fractional clock rate offset vs. orbital radius in the Schwarzschild spacetime.

    The curve shows dτ_sat/dτ_ground − 1 as a function of orbital radius from
    Earth's surface to ten Earth radii. A satellite clock runs fast relative to
    a ground clock because the gravitational blueshift — clocks tick faster
    further from a mass — outweighs the kinematic redshift from orbital
    velocity. A vertical line marks the GPS orbital radius; a horizontal line
    marks the ICD-GPS-200 reference correction factor.
    """
    M_geom = WGS84_MU / WGS84_C**2
    v_rot = WGS84_OMEGA_E * WGS84_R_E
    dtau_ground = math.sqrt((1 - 2 * M_geom / WGS84_R_E) - (v_rot / WGS84_C) ** 2)

    r = np.linspace(WGS84_R_E, 10 * WGS84_R_E, 2000)
    shift = np.sqrt(np.clip(1 - 3 * M_geom / r, 0, None)) / dtau_ground - 1

    fig, ax = plt.subplots()
    ax.plot(r / WGS84_R_E, shift)
    ax.axvline(GPS_SEMI_MAJOR_AXIS / WGS84_R_E, color="k", linestyle="--")
    ax.axhline(ICD_GPS200_FRACTIONAL_OFFSET, color="r", linestyle="--")
    ax.set_xlabel("r / R_E")
    ax.set_ylabel("Δf / f")
    plt.close(fig)
    return fig


__all__ = ["time_dilation_figure"]
