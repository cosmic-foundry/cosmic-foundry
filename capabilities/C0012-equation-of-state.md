# Capability: Equation of State

- **ID:** C0012
- **Status:** Proposed

## Abstract map

**Signature:** (ρ, T) → (P, e, s, c_v, c_s)

**Claim:** All thermodynamic quantities are determined by the specific
Helmholtz free energy F(ρ, T) via:

- P = ρ² ∂F/∂ρ|_T
- e = F − T ∂F/∂T|_ρ
- s = −∂F/∂T|_ρ
- c_v = −T ∂²F/∂T²|_ρ
- c_s = √(∂P/∂ρ|_s)

The map is exact. Θ = ∅.

**External reference:** Landau & Lifshitz, *Statistical Physics* (3rd ed.),
§15 (thermodynamic potentials and their derivatives).

## Conforming implementations

### Gamma-law

- **Θ:** ∅
- **p:** ∞ (exact)
- **Reason for Θ:** none — F(ρ, T) has a closed form
- **F(ρ, T):** c_v T [1 − ln(T^{1/γ−1} / ρ)] (up to constants)
- **Derived quantities:** P = (γ−1)ρe, c_s = √(γP/ρ)
- **Convergence evidence:** exact; verify against analytic identities
  (e.g. c_s² = γP/ρ to float64 rounding error)
- **External reference:** same as abstract map

### Tabulated Helmholtz (degenerate stellar matter)

- **Θ:** {h_ρ, h_T} — table spacing in log ρ and log T
- **p:** 4 (bicubic interpolation)
- **Reason for Θ:** F(ρ, T) for degenerate electrons involves
  Fermi-Dirac integrals; the closed form is exact but too expensive
  to evaluate per cell per timestep
- **Convergence evidence:** compare table output to direct
  Fermi-Dirac integration at a sequence of (h_ρ, h_T) values;
  verify O(h⁴) convergence in both directions
- **External reference:** Timmes & Swesty (2000), ApJS 126:501–516

## Open questions

- The (ρ, T) input convention assumed here; some fluid solvers carry
  (ρ, e) and must invert T = T(ρ, e). Whether that inversion is a
  separate map or a second signature of this one is unresolved.
- Additional output quantities needed by the reaction-network
  integrator (∂P/∂ρ|_e, ∂P/∂e|_ρ, etc.) — whether these extend
  the signature or define a child map.
