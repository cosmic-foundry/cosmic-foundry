"""GPS time dilation: Schwarzschild derivation validated against ICD-GPS-200.

Two claims:

1. Algebraic: the proper time ratio for a circular equatorial geodesic in the
   Schwarzschild spacetime simplifies to sqrt(1 - 3M/r). Verified by SymPy.

2. Numerical: the first-order weak-field fractional frequency shift evaluated
   at GPS orbital parameters matches the ICD-GPS-200 Table 20-IV correction
   factor of 4.4647 × 10⁻¹⁰ to within 1%.

   Omissions in this model (each < 1 × 10⁻¹² in magnitude):
   - Earth's rotation (Sagnac correction)
   - Second-order post-Newtonian terms
"""

from __future__ import annotations

import sympy

from validation.schwarzschild.spacetime import M, SchwarzschildSpacetime, r, theta


def test_circular_geodesic_proper_time() -> None:
    """Proper time ratio for circular equatorial orbit is sqrt(1 - 3M/r).

    Derived from the metric: dτ²/dt² = -g_tt - g_φφ · (dφ/dt)²
    with the circular geodesic condition (dφ/dt)² = M/r³.
    """
    metric = SchwarzschildSpacetime().metric

    g_tt = metric.component(0, 0).expr  # -(1 - 2M/r)
    g_phiphi = metric.component(3, 3).expr  # r² sin²θ

    g_phiphi_eq = g_phiphi.subs(theta, sympy.pi / 2)  # equatorial: → r²

    # Circular geodesic angular velocity in geometric units (G = c = 1)
    omega_sq = M / r**3

    dtau_sq = -g_tt - g_phiphi_eq * omega_sq

    assert sympy.simplify(dtau_sq - (1 - 3 * M / r)) == 0


def test_gps_clock_correction_matches_icd() -> None:
    """First-order fractional frequency shift matches ICD-GPS-200 within 1%.

    The weak-field expansion of (dτ_sat/dτ_ground - 1) to first order in M/r
    gives M·(1/r_E - 3/(2r_gps)). Evaluated at WGS 84 parameters this agrees
    with the ICD-GPS-200 Table 20-IV value of 4.4647 × 10⁻¹⁰.
    """
    r_E = sympy.Symbol("r_E", positive=True)
    r_gps = sympy.Symbol("r_gps", positive=True)

    # Proper time rates from the geodesic result above
    dtau_sat = sympy.sqrt(1 - 3 * M / r)
    dtau_ground = sympy.sqrt(1 - 2 * M / r_E)

    shift = dtau_sat.subs(r, r_gps) / dtau_ground - 1

    # Linearise in M (weak-field; M/r ~ 7e-10 at GPS altitude)
    shift_linear = sympy.series(shift, M, 0, 2).removeO()

    expected_formula = M * (1 / r_E - sympy.Rational(3, 2) / r_gps)
    assert sympy.simplify(shift_linear - expected_formula) == 0

    # Numerical evaluation — WGS 84 / ICD-GPS-200 constants
    mu_E = 3.986005e14  # m³ s⁻²  standard gravitational parameter
    c_si = 2.99792458e8  # m s⁻¹
    r_E_si = 6.3781e6  # m       mean Earth radius
    r_gps_si = 2.6560e7  # m       GPS nominal semi-major axis

    M_si = mu_E / c_si**2  # geometric mass [m]

    shift_num = float(
        expected_formula.subs([(M, M_si), (r_E, r_E_si), (r_gps, r_gps_si)])
    )

    # ICD-GPS-200 Table 20-IV: satellite clocks run fast by 4.4647 × 10⁻¹⁰
    assert abs(shift_num - 4.4647e-10) / 4.4647e-10 < 0.01
