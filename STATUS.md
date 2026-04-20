# Cosmic Foundry — Status

The repository is in a foundation-building phase. We have cleared away wrong
architectural assumptions (the old Op/Region/Policy framing, the `geometry/`
domain package, `computation/Array` and `Extent` wrappers, the CLI entry point)
and rebuilt on a clean mathematical foundation.

## What is complete

**`theory/`** — pure mathematical ABCs, all tested (96 passing):

- Set hierarchy: `Set`, `IndexedSet`, `IndexedFamily`, `Function`
- Manifold hierarchy: `Manifold` → `SmoothManifold` → `PseudoRiemannianManifold`
  → `RiemannianManifold` / `FlatManifold` → `EuclideanSpace` / `MinkowskiSpace`
- `ManifoldWithBoundary` → `Region`
- Field hierarchy: `Field` → `TensorField` → `VectorField`, `SymmetricTensorField`
  → `MetricTensor`, `DifferentialForm` → `ScalarField` / `CovectorField`
- `DifferentialOperator`
- `BoundaryCondition` → `LocalBoundaryCondition` / `NonLocalBoundaryCondition`

**`derivations/`** — SymPy-based finite-difference stencil coefficient derivation
for arbitrary derivative order and approximation order, with convergence tests.

## What is not yet started

`computation/` is empty. There are no concrete implementations of any `theory/` ABC:
no JAX-backed fields, no grids, no numerical operators, no I/O.

## Near-term work

**Review and sharpen the long-horizon epoch sequence in `ROADMAP.md`.**

The roadmap was written before the `theory/` layer existed. Its epoch sequence
jumps from early scaffolding to AMR mesh without accounting for the concrete
implementation steps that bridge the abstract layer to a working physics code.
The next session should establish a realistic sequence from theory → concrete
geometry → concrete fields → numerical operators → first physics, and sharpen
the remaining epochs accordingly.
