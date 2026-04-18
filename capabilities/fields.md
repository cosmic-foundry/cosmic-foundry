# Fields

The fields we are attempting to model. Every map signature is expressed
in terms of entries from this table.

A field is a function on a domain. Rank specifies tensor rank: 0 = scalar,
1 = vector, 2 = tensor.

The domain for all fields below is Ω × T unless noted, where Ω ⊆ ℝ³ is
the spatial domain and T ⊆ ℝ is the time interval.

Derivative fields: ∂_t or ∂_x applied to any listed field produces another
field of the same rank. These are not listed separately — every listed field
implicitly has corresponding derivative fields that appear as intermediate
arrays in the code.

## Fluid state

| Symbol | Name | Rank | Physical meaning |
|--------|------|------|-----------------|
| ρ | mass density | 0 | mass per unit volume |
| **v** | velocity | 1 | bulk fluid velocity |
| e | specific internal energy | 0 | thermal energy per unit mass |

## Thermodynamic quantities

| Symbol | Name | Rank | Physical meaning |
|--------|------|------|-----------------|
| P | pressure | 0 | isotropic mechanical stress |
| T | temperature | 0 | thermodynamic temperature |
| s | specific entropy | 0 | entropy per unit mass |
| c_v | specific heat at constant volume | 0 | ∂e/∂T\|_ρ |
| c_s | adiabatic sound speed | 0 | √(∂P/∂ρ\|_s) |

## Gravitational fields

| Symbol | Name | Rank | Physical meaning |
|--------|------|------|-----------------|
| φ | gravitational potential | 0 | potential energy per unit mass |

## Nuclear composition

| Symbol | Name | Rank | Physical meaning |
|--------|------|------|-----------------|
| X_i | mass fraction of species i | 0 | fraction of mass in nuclear species i; Σ X_i = 1 |
