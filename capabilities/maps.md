# Maps

The maps we are attempting to implement. Every entry states what the map
computes exactly — no implementation choices, no approximation parameters.

A map is a relationship between fields. The claim for each map is the
exact mathematical statement that a conforming implementation must
approximate.

## Equation of state

**Signature:**

| Inputs | Outputs | Generating potential | Primary use |
|--------|---------|---------------------|-------------|
| (ρ, T) | (P, e, s, c_v, c_s) | Helmholtz F(ρ, T) | tabulated EOS |
| (ρ, e) | (P, T, s, c_v, c_s) | internal energy U(ρ, s) inverted | conservative hydro |
| (ρ, s) | (P, T, e, c_v, c_s) | internal energy U(ρ, s) | isentropic flows |

**Claim:** All outputs follow from the generating potential by
differentiation. For the (ρ, T) signature with F(ρ, T) the specific
Helmholtz free energy:

- P = ρ² ∂F/∂ρ|_T
- e = F − T ∂F/∂T|_ρ
- s = −∂F/∂T|_ρ
- c_v = −T ∂²F/∂T²|_ρ
- c_s = √(∂P/∂ρ|_s)

Equivalent relations hold for the other signatures via their respective
generating potentials.

**External reference:** Landau & Lifshitz, *Statistical Physics* (3rd ed.),
§15; Callen, *Thermodynamics and an Introduction to Thermostatistics*
(2nd ed.), §1.3.

## Poisson equation

**Signature:** ρ → φ

**Claim:** ∇²φ = 4πGρ

**External reference:** Poisson (1813), *Bulletin de la Société
Philomathique*, p. 388.

## Euler equations

**Signature:** (ρ, **v**, e, P, φ) → ∂_t(ρ, ρ**v**, ρE)

**Claim:** The conserved quantities (mass, momentum, total energy) evolve
according to:

- ∂_t ρ + ∇·(ρ**v**) = 0
- ∂_t(ρ**v**) + ∇·(ρ**v** ⊗ **v** + P**I**) = −ρ∇φ
- ∂_t(ρE) + ∇·((ρE + P)**v**) = −ρ**v**·∇φ

where E = e + ½|**v**|² is the specific total energy.

**External reference:** Euler (1757), *Mémoires de l'Académie des Sciences
de Berlin* 11:274–315; Landau & Lifshitz, *Fluid Mechanics* (2nd ed.), §1.

## Nuclear reaction network

**Signature:** (X_i, R_{ij}) → (∂_t X_i, ε)

**Claim:**

∂_t X_i = (A_i / N_A) Σ_j R_{ij}

ε = −Σ_i (∂_t X_i) Q_i / A_i

where A_i is the atomic mass of species i, N_A is Avogadro's number,
and Q_i is the binding energy per nucleon of species i.

**External reference:** Arnett & Truran (1969), ApJ 157:339; Timmes (1999),
ApJS 124:241.

## Reaction rates

**Signature:** (ρ, T) → R_{ij}

**Claim:** Each rate R_{ij}(ρ, T) is determined by nuclear physics: a
combination of resonant and non-resonant contributions integrated over
the Maxwell-Boltzmann velocity distribution. For two-body reactions:

R_{ij} = ρ N_A ⟨σv⟩_{ij}(T)

where ⟨σv⟩ is the thermally averaged reaction cross-section.

**External reference:** Iliadis, *Nuclear Physics of Stars* (2nd ed.), §3;
NACRE II (Xu et al. 2013, NPA 918:61) and REACLIB (Cyburt et al. 2010,
ApJS 189:240) for tabulated rate data.
