# Capability: Forward gravitational field evaluation

- **ID:** C0010
- **Status:** Proposed
- **Implemented in:** not yet

## Behavior

Given a mass distribution for which a closed-form relationship between the
distribution and its gravitational field is known, evaluate the gravitational
field (potential and/or acceleration) at arbitrary points in 3D space.

Conforming implementations include:

- Point mass: `g = -GM/r²` directed toward the source
- Shell theorem: a spherically symmetric body is equivalent to a point mass
  at its center for exterior points
- Multipole expansion: the field is expressed as a series in spherical
  harmonics; each order is evaluated analytically given the multipole
  moments of the distribution

All implementations share the same interface and must pass the same
verification tests.

## Correctness claim

For any test case where the exact field is known analytically, the
evaluated field must agree with the analytical answer to within float64
rounding error (no discretization error — these are closed-form
evaluations, not numerical approximations).

## External grounding

- **Point mass / shell theorem:** `g = -GM/r²` is Newton's law of
  gravitation. Verified against the analytical field of a uniform sphere
  at exterior points (shell theorem exact result).
- **Multipole expansion:** each order l verified against the analytical
  field of a mass distribution whose multipole moments are known exactly
  (e.g. uniform sphere: only l=0 nonzero; uniform ellipsoid: l=0 and l=2
  nonzero with analytically known coefficients).

The same analytical test cases used here serve as the external grounding
for C0011 (backward solver) — agreement between forward and backward on
these cases is an additional verification criterion for C0011.

## Open questions

- How many multipole orders are needed for the WD inspiral phase before
  the Poisson solver must take over?
- Truncation error of the multipole series as a function of separation
  and order — needs a convergence spec once implementation begins.
