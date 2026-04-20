# Cosmic Foundry — Status

The repository is organized around a three-layer architecture:

- **Continuous** (`theory/`) — manifolds, fields, operators, boundary conditions.
  The problem stated in its true mathematical form.
- **Discrete** — a chosen discretization: grid, scheme, stencils, discrete fields.
  Still symbolic; no floating-point arrays.
- **Numerical** (`computation/`) — JAX evaluates the discrete description.

## What is complete

**Foundation layer (`foundation/`)** — 104 tests passing:

- `Set`, `Function`, `IndexedSet`, `IndexedFamily`

**Continuous layer (`continuous/`)** — all ABCs tested:

- Manifold hierarchy: `Manifold` → `SmoothManifold` → `PseudoRiemannianManifold`
  → `RiemannianManifold` / `FlatManifold` → `EuclideanSpace` / `MinkowskiSpace`
- `ManifoldWithBoundary` → `Region`
- Field hierarchy: `Field` → `TensorField` → `VectorField`, `SymmetricTensorField`
  → `MetricTensor`, `DifferentialForm` → `ScalarField` / `CovectorField`
- `DifferentialOperator`
- `BoundaryCondition` → `LocalBoundaryCondition` / `NonLocalBoundaryCondition`

**Discrete layer (`discrete/`)** — ABCs only:

- `DiscreteField`, `DiscreteScalarField`, `DiscreteVectorField` with typed
  `approximates: Optional[<continuous counterpart>]` property

## What is not yet started

No concrete discrete implementations: no grid, no stencil operators, no
JAX-backed evaluation, no I/O.

## Near-term work

**Review and sharpen the epoch sequence in `ROADMAP.md`.** The roadmap has been
restructured around the continuous/discrete/numerical split, but the epoch
boundaries — particularly the shape of the discrete layer and where JAX first
enters — need a design session before implementation begins.
