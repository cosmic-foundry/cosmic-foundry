"""WGS 84 / IS-GPS-200 physical constants.

Source: IS-GPS-200 Rev K, Interface Specification for GPS (GPS Directorate).
https://www.gps.gov/technical/icwg/IS-GPS-200K.pdf  Table 20-IV.
"""

# Pre-launch fractional frequency offset applied to GPS satellite clocks.
# Satellite clocks run fast relative to ground; clocks are manufactured slow.
ICD_GPS200_FRACTIONAL_OFFSET: float = 4.4647e-10  # dimensionless

WGS84_MU: float = 3.986005e14  # m³ s⁻²   Earth standard gravitational parameter
WGS84_OMEGA_E: float = 7.2921151467e-5  # rad s⁻¹  Earth rotation rate
WGS84_R_E: float = 6.378137e6  # m         equatorial radius
WGS84_C: float = 2.99792458e8  # m s⁻¹    speed of light
GPS_SEMI_MAJOR_AXIS: float = 26_559_710.0  # m       nominal GPS orbital radius

__all__ = [
    "GPS_SEMI_MAJOR_AXIS",
    "ICD_GPS200_FRACTIONAL_OFFSET",
    "WGS84_C",
    "WGS84_MU",
    "WGS84_OMEGA_E",
    "WGS84_R_E",
]
