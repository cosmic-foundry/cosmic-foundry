# ADR-0016 — Field / function formalism

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
Operator classes each transformed one thing into another, but their domain /
codomain / approximation properties were implicit. New contributors had no
formal basis for deciding whether a new class should be a field, a function,
or something else.

The underlying issue was that the codebase lacked a two-concept organizing
principle: an answer to "what is the fundamental object?" and an answer to
"what is the relationship between objects?"

## Decision

**Set, field, function, source, and sink are the five primitive organizing concepts.**

### Set

A set is a collection of elements. `Set` is the root of the mathematical
type hierarchy. Two disjoint branches descend from it:

- `IndexedSet(Set)` — a finite set with a bijection to a subset of ℤⁿ;
  carries abstract `ndim` and `shape`. Concrete: `Patch`.
- `SmoothManifold(Set)` — a set with a smooth structure enabling calculus;
  carries abstract `ndim`. Extends to `PseudoRiemannianManifold` and
  `RiemannianManifold`.

### Field

A field is a function on a manifold M with values in a vector space V:

```
f : M → V
```

`Field(Function)` is the ABC for all fields. Subclasses specialize by codomain:

| Class | Codomain V | Nature |
|-------|------------|--------|
| `ScalarField(Field)` | ℝ | Marker; `execute` remains abstract |
| `TensorField(Field)` | T^(p,q)M | Abstract `tensor_type: tuple[int, int]` |

Concrete scalar fields:

| Class | Domain M | Parameter space Θ | Representation |
|-------|----------|-------------------|----------------|
| `ContinuousField(ScalarField)` | M (any manifold; exact) | ∅ | A Python callable |
| `PatchFunction(ScalarField)` | Ω_h (discrete grid patch) | {h} (grid spacing) | JAX array payload |

`ContinuousField` carries no approximation: evaluating it at any point in M
returns the exact field value. `PatchFunction` carries approximation error
O(hᵖ) for smooth fields; p depends on the discretization scheme that
produced the payload.

`PatchFunction` is the data-side counterpart of `Patch` (the geometry).
`Patch` records topology and node positions; `PatchFunction` records the
field values at those nodes. Both are frozen dataclasses and independently
composable.

**Domain M is unrestricted for `ContinuousField`.** M may be physical space
(Ω ⊆ ℝⁿ), thermodynamic state space ((ρ, T)), or any other manifold on
which an exact callable is meaningful. An equation of state f: (ρ, T) → P
is a `ContinuousField` on thermodynamic state space. Tabulating it onto a
discrete (ρ, T) grid is a `discretize` call — the same operation as
sampling a spatial field onto a simulation grid, with M = thermodynamic
state space instead of M = Ω ⊆ ℝⁿ.

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

Sinks differ from functions in that their output is not a value in a
mathematical codomain but a side effect on the world outside the computation.
Every non-trivial sink carries a ``Sink:`` block in its docstring:

```
Sink:
    domain — description of the input consumed
    effect — external state produced or modified
```

Free functions that are sinks carry the ``Sink:`` block in the function
docstring; sink classes carry it in the class docstring.

### Function

A function is a relationship between mathematical objects:

```
f : A × Θ → B
```

where A is the domain (one or more fields or grids), B is the codomain
(a field or scalar), Θ is the parameter space, and p is the approximation
order. Θ = ∅ denotes an exact function; Θ = {h, …} denotes parameters
that govern approximation error.

**Every non-trivial operator class is a function.** A class that transforms
inputs into outputs — whether by evaluating a kernel, filling ghost cells,
or reducing a field to a scalar — must be describable precisely as a function.
If a class cannot be given a well-defined domain, codomain, and operator,
it should be redesigned until it can.

`Function`, `Source`, and `Sink` are abstract base classes.  Every concrete
function/source/sink class inherits from the appropriate ABC and implements
`execute()`.  All three ABCs provide a default `__call__` that delegates to
`execute()`, so instances are directly callable.  Stateless instances
(Θ = ∅, no constructor parameters) expose a module-level singleton for
convenience: `collect_diagnostics = CollectDiagnostics()`,
`global_sum = GlobalSum()`, `write_array = WriteArray()`,
`merge_rank_files = MergeRankFiles()`.

Desiderata are not function parameters. A consumer's requirement on a
function's output (e.g. "I need N ghost cells") is a constraint on which
function to call, not a parameter of the function itself. It must not appear
in the function's signature.

### Docstring convention

Every operator class carries a `Function:` block in its class docstring:

```
Function:
    domain   — description of the input type(s)
    codomain — description of the output type
    operator — informal description of what the operator does

Θ = {…} — description of approximation parameters, or ∅ if exact.
p = N   — approximation order, when Θ ≠ ∅.
```

Free functions that are functions (e.g. `GlobalSum`, `CollectDiagnostics`)
carry the `Function:` block in the class docstring.

### Canonical concrete discretizations

| Class | Nature | ndim |
|---|---|---|
| `Patch` | Ω_h — one contiguous patch of uniformly-spaced cells | spatial dimension |
| `Array[Patch]` | Partitioned spatial domain — a finite indexed family of Patches | spatial dimension |

### Canonical function instances

| Class / function | Domain | Codomain | Θ |
|---|---|---|---|
| `partition_domain` | Ω × ℤⁿ (n_cells, blocks) | `Array[Patch]` | {h} |
| `discretize` | `ContinuousField` × `Array[Patch]` | `Array[PatchFunction]` | {h}, p = 1 |
| `fill_halo` | `Array[PatchFunction]` × `AccessPattern` | `Array[PatchFunction]` with filled halos | ∅ |
| `GlobalSum` | `Array[PatchFunction]` × `Region` | scalar | ∅ |
| `CollectDiagnostics` | reducers × fields × `Region` | `DiagnosticRecord` | ∅ |

## Consequences

- **Positive.** The five-concept vocabulary (set, field, function, source, sink)
  provides a decision procedure for every new class: is this a set
  (a mathematical collection), a field (a function on a manifold), a function
  (a transformation returning a mathematical object), a source (a read from
  external state), or a sink (a write to external state)? The `Function:`,
  `Source:`, and `Sink:` docstring conventions make approximation properties
  and I/O contracts explicit and machine-auditable. Desiderata are cleanly
  separated from operator signatures.
- **Negative.** Every operator class must be written with a `Function:` block.
  The requirement adds authoring overhead for new contributors. Classes that
  resist a clean function description reveal a design problem that must be
  resolved before merging, which may slow development.
- **Neutral.** `ContinuousField` and `PatchFunction` are distinct classes;
  functions that accept either must use the `Field` ABC. Functions that apply
  only to `PatchFunction` (e.g. `fill_halo`) use `PatchFunction` directly
  rather than the ABC, because the ABC provides only `name: str` and the
  abstract `execute`.

## Alternatives considered

**One `Field` class covering both representations.** A single `Field` class
with an optional callable and optional array segments would avoid the
`ContinuousField` / `PatchFunction` split. Rejected because the two
representations have incompatible parameter spaces (Θ = ∅ vs. Θ = {h}) and
incompatible domains (M vs. Ω_h). Unifying them obscures the approximation
relationship that is the central concern of numerical methods.

**`AccessPattern` as a parameter of `discretize`.** The original
`allocate_field` accepted an `AccessPattern` to size ghost-cell halos. Rejected
because ghost-cell width is a property of the downstream consumer (an
operator's stencil), not of the discretization of a continuous field. Including
it in `discretize` would make the function's output depend on a future
consumer's requirements, violating the clean domain → codomain structure.
Ghost-cell storage belongs to a separate function (`fill_halo`) whose domain
is the consumer's stencil requirements, not the source field.

**Informal conventions without an ADR.** The `Function:`, `Source:`, and
`Sink:` block patterns could be treated as style guide items rather than an
architectural decision. Rejected because without a recorded decision, new
contributors have no basis for understanding why `AccessPattern` does not
belong in `discretize`, what "Θ = ∅" means in context, why `write_array` has
a `Sink:` block and `load_schema` has a `Source:` block instead of a
`Function:` block, or that EOS tabulation is a `discretize` call on
thermodynamic state space rather than a different kind of thing.
The formalism is the architecture; the ADR is the primary explanation of it.
