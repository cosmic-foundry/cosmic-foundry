# Capability: Gravitational potential

- **ID:** C0013
- **Status:** Proposed

## Abstract map

**Signature:** ρ → φ

**Claim:** Given a mass density field ρ on a domain Ω ⊆ ℝ³, the
gravitational potential φ is the unique solution to:

∇²φ = 4πGρ,   φ(r) → 0 as |r| → ∞

Equivalently, via Green's function:

φ(r) = −G ∫ ρ(r') / |r − r'| d³r'

The map is exact. Θ = ∅.

**External reference:** Newton (1687), *Principia*, Book I, Propositions
71–75 (shell theorem); Poisson (1813), *Bulletin de la Société
Philomathique*, p. 388 (field equation).

## Conforming implementations

### Point mass

- **Θ:** ∅
- **p:** ∞ (exact)
- **Reason for Θ:** none — closed form for a mass concentrated at a point
- **Map:** φ(r) = −GM/|r − r₀|
- **Convergence evidence:** exact; verify to float64 rounding error
  against the analytical expression at arbitrary r
- **Validity:** exact only when ρ is a point mass; otherwise an
  approximation whose error is O((a/r)²) where a is the source size

### Shell theorem (spherically symmetric distribution)

- **Θ:** ∅
- **p:** ∞ (exact)
- **Reason for Θ:** none — closed form for spherically symmetric ρ(r)
- **Map:** for r > R (exterior), φ(r) = −GM/r;
  for r < R (interior of uniform shell), φ = const
- **Convergence evidence:** exact; verify exterior field of a uniform
  sphere against −GM/r to float64 rounding error
- **External reference:** Newton (1687), Proposition 71

### Multipole expansion

- **Θ:** {l_max} — maximum multipole order retained
- **p:** error ~ O((a/r)^{l_max+1}) in the far field
- **Reason for Θ:** the full multipole series is exact; truncation at
  l_max introduces error that decreases with distance and order
- **Map:** φ(r) = −G Σ_{l=0}^{l_max} Σ_m Q_lm Y_lm(r̂) / r^{l+1}
  where Q_lm are the multipole moments of ρ
- **Convergence evidence:** for a uniform ellipsoid (l=0,2 moments
  known analytically), verify convergence of the series as l_max
  increases against the analytical exterior potential
- **External reference:** Jackson, *Classical Electrodynamics* (3rd ed.),
  §3.4 (multipole expansion; gravitational case by analogy)

### FFT with zero-padding

- **Θ:** {h} — grid spacing
- **p:** 2
- **Reason for Θ:** spectral solve on a discrete grid; zero-padding
  enforces isolated boundary conditions without periodic artifacts
- **Convergence evidence:** MMS — choose φ_exact with known ∇²φ_exact,
  verify O(h²) convergence of the recovered potential
- **External reference:** Hockney & Eastwood (1988), *Computer Simulation
  Using Particles*, §6.4 (FFT Poisson with isolated BCs)

### James method

- **Θ:** {h} — grid spacing
- **p:** 2
- **Reason for Θ:** finite-difference interior solve corrected by a
  Green's function screening charge on the boundary
- **Convergence evidence:** same MMS test as FFT; additionally verify
  agreement with FFT implementation on uniform-sphere test to within
  O(h²)
- **External reference:** James (1977), *Journal of Computational
  Physics* 25:71–93; Mattia & Vignoli (2019), ApJS 241:26

### Multipole boundary matching

- **Θ:** {h, l_max}
- **p:** min(2, O((a/r)^{l_max+1})) — limited by whichever error dominates
- **Reason for Θ:** finite-difference interior solve with boundary values
  supplied by a truncated multipole expansion of the interior mass
- **Convergence evidence:** verify O(h²) convergence at fixed l_max on
  a smooth test case; separately verify convergence in l_max at fixed h

## Open questions

- For the multipole expansion and multipole boundary matching, the
  transition criterion between l_max = 0 (monopole) and higher orders
  is problem-dependent — needs a spec for the WD inspiral phase.
- All three discrete implementations (FFT, James, multipole matching)
  must agree to within O(h²) on shared test cases; the cross-check
  protocol is not yet defined.
