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

A field is a function on a domain:

```
f : D → ℝ
```

A field is not a map. It is the mathematical object that maps are defined over.
The `Field` ABC has two concrete parameterizations:

| Class | Domain D | Parameter space Θ | Representation |
|-------|----------|-------------------|----------------|
| `ContinuousField` | D (any domain; exact) | ∅ | A Python callable |
| `DiscreteField` | D_h ⊂ D (discrete grid) | {h} (grid spacing) | Payload arrays per block |

`ContinuousField` carries no approximation: evaluating it at any point in D
returns the exact field value. `DiscreteField` carries approximation error
O(hᵖ) for smooth fields; p depends on the discretization scheme.

**Domain D is unrestricted.** `ContinuousField` represents any exact scalar
field f: D → ℝ regardless of what D is. D may be physical space (Ω ⊆ ℝⁿ),
thermodynamic state space ((ρ, T)), or any other domain on which an exact
callable is meaningful. An equation of state f: (ρ, T) → P is a
`ContinuousField` on thermodynamic state space. Tabulating it onto a discrete
(ρ, T) grid is a `FieldDiscretization` — the same operation as sampling a
spatial field onto a simulation grid, with D = thermodynamic state space
instead of D = Ω ⊆ ℝⁿ.

### Source

A source reads from external state — a file, a network endpoint, a device —
and returns a mathematical object or data structure:

```
R : external state → B
```

Sources are the read-side complement of sinks. Every non-trivial source
carries a ``Source:`` block in its docstring:

```
Source:
    origin   — external state consumed (file path, URL, stream, etc.)
    produces — mathematical object or data structure returned
```

### Sink

A sink is an operation that consumes data and materialises it into external
state — a file, a network message, a display — rather than returning a
mathematical object:

```
S : A → external state
```

Sinks are the third primitive alongside fields and maps. They differ from
maps in that their output is not a value in a mathematical codomain but a
side effect on the world outside the computation. Every non-trivial sink
carries a ``Sink:`` block in its docstring:

```
Sink:
    domain — description of the input consumed
    effect — external state produced or modified
```

Free functions that are sinks carry the ``Sink:`` block in the function
docstring; sink classes carry it in the class docstring.

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

`Map`, `Source`, and `Sink` are abstract base classes in
`cosmic_foundry.kernels`.  Every concrete map/source/sink class inherits
from the appropriate ABC and implements `execute()`.  All three ABCs
provide a default `__call__` that delegates to `execute()`, so instances
are directly callable.  Stateless instances (Θ = ∅, no constructor
parameters) expose a module-level singleton for convenience:
`collect_diagnostics = CollectDiagnostics()`, `global_sum = GlobalSum()`,
`write_array = WriteArray()`, `merge_rank_files = MergeRankFiles()`,
`load_schema = LoadSchema()`.

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
| `GlobalSum` | `DiscreteField` × `Region` | scalar | ∅ |
| `CollectDiagnostics` | reducers × fields × `Region` | `DiagnosticRecord` | ∅ |

## Consequences

- **Positive.** The four-concept vocabulary (field, map, source, sink)
  provides a decision procedure for every new class: is this a field
  (a function on a domain), a map (a transformation returning a mathematical
  object), a source (a read from external state), or a sink (a write to
  external state)? The `Map:`, `Source:`, and `Sink:` docstring conventions
  make approximation properties and I/O contracts explicit and
  machine-auditable. Desiderata are cleanly separated from operator
  signatures.
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

**Informal conventions without an ADR.** The `Map:`, `Source:`, and `Sink:`
block patterns could be treated as style guide items rather than an
architectural decision. Rejected because without a recorded decision, new
contributors have no basis for understanding why `AccessPattern` does not
belong in `FieldDiscretization`, what "Θ = ∅" means in context, why
`write_array` has a `Sink:` block and `load_schema` has a `Source:` block
instead of a `Map:` block, or that EOS tabulation is a `FieldDiscretization`
on thermodynamic state space rather than a different kind of thing.
The formalism is the architecture; the ADR is the primary explanation of it.
