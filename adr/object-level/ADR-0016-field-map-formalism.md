# ADR-0016 — Field / map formalism

## Context

The codebase accumulated several classes — `allocate_field`, `FieldDiscretization`,
`HaloFillPolicy`, `FlatPolicy`, `global_sum` — whose relationships to one another
were described only informally. Two symptoms forced a reckoning:

**1. `allocate_field` conflated distinct operations.**
The function accepted an `AccessPattern` (a desideratum about ghost-cell
width) alongside mesh topology, combined memory reservation, zero-initialization,
and discretization into one call, and described the result as "storage." But
storage parameterized by a stencil width is not a well-defined mathematical
object: the stencil is the consumer's concern, not the discretization operator's.

**2. No shared language for what operator classes produce.**
`FlatPolicy.execute`, `HaloFillPolicy.execute`, and `global_sum` each transformed
one thing into another, but their domain / codomain / approximation properties
were implicit. New contributors had no formal basis for deciding whether a new
class should be a field, a map, or something else.

The underlying issue was that the codebase lacked a two-concept organizing
principle: an answer to "what is the fundamental object?" and an answer to
"what is the relationship between objects?"

## Decision

**Fields and maps are the two primitive organizing concepts.**

### Field

A field is a function on a spatial domain:

```
f : D → ℝ
```

A field is not a map. It is the mathematical object that maps are defined over.
The `Field` ABC has two concrete parameterizations:

| Class | Domain D | Parameter space Θ | Representation |
|-------|----------|-------------------|----------------|
| `ContinuousField` | Ω ⊆ ℝⁿ (continuous) | ∅ (exact) | A Python callable |
| `DiscreteField` | Ω_h ⊂ Ω (grid points) | {h} (grid spacing) | Array segments over `FieldSegment`s |

`ContinuousField` carries no approximation: evaluating it at any point in Ω
returns the exact field value. `DiscreteField` carries approximation error
O(hᵖ) for smooth fields; p depends on the discretization scheme.

**Scope of `ContinuousField`.** `ContinuousField` represents fields on physical
space: f: Ω ⊆ ℝⁿ → ℝ evaluated at spatial coordinates. It does not represent
maps on other state spaces. An equation of state, for example, is a map on
thermodynamic state space (ρ, T) → (P, e, …) — not a `ContinuousField`, because
its domain is not a spatial domain.

### Map

A map is a relationship between fields:

```
M : A × Θ → B
```

where A is the domain (one or more fields or grids), B is the codomain
(a field or scalar), Θ is the parameter space, and p is the approximation order.
Θ = ∅ denotes an exact map; Θ = {h, …} denotes parameters that govern
approximation error.

**Every non-trivial operator class is a map.** A class that transforms
inputs into outputs — whether by evaluating a kernel, filling ghost cells,
or reducing a field to a scalar — must be describable precisely as a map.
If a class cannot be given a well-defined domain, codomain, and operator,
it should be redesigned until it can.

Desiderata are not map parameters. A consumer's requirement on a map's
output (e.g. "I need N ghost cells") is a constraint on which map to call,
not a parameter of the map itself. It must not appear in the map's signature.

### Docstring convention

Every operator class carries a `Map:` block in its class docstring:

```
Map:
    domain   — description of the input type(s)
    codomain — description of the output type
    operator — informal description of what the operator does

Θ = {…} — description of approximation parameters, or ∅ if exact.
p = N   — approximation order, when Θ ≠ ∅.
```

Free functions that are maps (e.g. `global_sum`, `collect_diagnostics`)
carry the `Map:` block in the function docstring.

### Canonical map instances

| Class / function | Domain | Codomain | Θ |
|---|---|---|---|
| `UniformGrid.create` | Ω × ℤⁿ (n_cells, blocks) | Ω_h (block partition) | {h} |
| `FieldDiscretization` | `ContinuousField` × `UniformGrid` | `DiscreteField` | {h}, p = 1 |
| `FlatPolicy` | `BoundOp` × `Region` | array over Ω_h^int | ∅ |
| `Dispatch` | `BoundOp` × `Region` × `FlatPolicy` | policy result | ∅ |
| `HaloFillPolicy` | `DiscreteField` × `Region` × `Stencil` | `DiscreteField` with filled halos | ∅ |
| `global_sum` | `DiscreteField` × `Region` | scalar | ∅ |
| `collect_diagnostics` | reducers × fields × `Region` | `DiagnosticRecord` | ∅ |

## Consequences

- **Positive.** The two-concept vocabulary (field, map) provides a decision
  procedure for every new class: is this a field (a function on a domain)
  or a map (a transformation)? The `Map:` docstring convention makes
  approximation properties explicit and machine-auditable. Desiderata are
  cleanly separated from operator signatures.
- **Negative.** Every operator class must be written with a `Map:` block.
  The requirement adds authoring overhead for new contributors. Classes that
  resist a clean map description reveal a design problem that must be resolved
  before merging, which may slow development.
- **Neutral.** `ContinuousField` and `DiscreteField` are distinct classes;
  functions that accept either must use the `Field` ABC. Maps that apply
  only to `DiscreteField` (e.g. `HaloFillPolicy`) use `DiscreteField`
  directly rather than the ABC, because the ABC provides only `name: str`.

## Alternatives considered

**One `Field` class covering both representations.** A single `Field` class
with an optional callable and optional array segments would avoid the
`ContinuousField` / `DiscreteField` split. Rejected because the two
representations have incompatible parameter spaces (Θ = ∅ vs. Θ = {h}) and
incompatible domains (Ω vs. Ω_h). Unifying them obscures the approximation
relationship that is the central concern of numerical methods.

**`AccessPattern` as a parameter of `FieldDiscretization`.** The original
`allocate_field` accepted an `AccessPattern` to size ghost-cell halos. Rejected
because ghost-cell width is a property of the downstream consumer (an Op's
stencil), not of the discretization of a continuous field. Including it in
`FieldDiscretization` would make the map's output depend on a future consumer's
requirements, violating the clean domain → codomain structure. Ghost-cell
storage belongs to a separate map (halo allocation) whose domain is the
consumer's stencil requirements, not the source field.

**Informal conventions without an ADR.** The `Map:` block pattern could be
treated as a style guide item rather than an architectural decision. Rejected
because without a recorded decision, new contributors have no basis for
understanding why `AccessPattern` does not belong in `FieldDiscretization`,
why `ContinuousField` does not model EOS, or what "Θ = ∅" means in context.
The formalism is the architecture; the ADR is the primary explanation of it.
