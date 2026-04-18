# Capability: Backward gravitational field recovery (Poisson solver)

- **ID:** C0011
- **Status:** Proposed
- **Implemented in:** not yet

## Behavior

Given an arbitrary mass distribution on a finite computational domain,
recover the gravitational potential by solving Poisson's equation:

```
∇²φ = 4πGρ
```

with isolated boundary conditions (the field falls off to zero at
infinity; the domain is finite).

Conforming implementations include:

- **FFT with zero-padding:** spectral solve on a zero-padded domain to
  enforce isolated BCs without periodic artifacts
- **James method:** interior solve with zero BCs corrected by a Green's
  function screening charge convolution (James 1977; extended to
  cylindrical coordinates by Mattia & Vignoli 2019,
  https://doi.org/10.3847/1538-4365/ab1a12)
- **Multipole boundary matching:** multipole expansion evaluated at the
  domain boundary to supply boundary values for an interior finite-difference
  or finite-volume solve

All implementations share the same interface and must pass the same
verification tests.

## Correctness claim

For any smooth density distribution whose exact gravitational potential
is known analytically, the recovered potential must converge to the
analytical answer at second order in the grid spacing h.

## External grounding

Verified against analytical test cases where the exact potential is known
from C0010 (forward evaluation):

- **Uniform sphere:** potential is quadratic inside, `φ ∝ 1/r` outside.
  Both interior and exterior must converge at second order.
- **Plummer sphere:** `φ = -GM / √(r² + a²)`, analytical potential known
  everywhere.

Agreement between all three conforming implementations on these test cases
is a required verification criterion — implementations must agree to within
discretization error, not just match the analytical answer independently.

## Open questions

- Domain size requirements for FFT zero-padding to achieve second-order
  isolated BCs without excessive memory cost.
- Whether James method or multipole matching is preferred for cylindrical
  geometry (relevant if the fluid solver uses cylindrical coordinates).
- Transition criterion from C0010 (forward) to C0011 (backward) during
  the WD inspiral — likely when tidal deformation makes the multipole
  series too expensive to converge.
