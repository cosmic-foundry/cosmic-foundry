# Cosmic Foundry — Status

The repository is organized around four packages:

- **Foundation** (`foundation/`) — `Set`, `Function`, `IndexedSet`, `IndexedFamily`.
- **Continuous** (`continuous/`) — manifolds, fields, operators, boundary conditions.
- **Discrete** (`discrete/`) — scheme description on finite index sets; symbolic.
- **Numerical** (`computation/`) — JAX evaluates the discrete description.

`foundation/`, `continuous/`, and `discrete/` are symbolic-reasoning layers:
no floats, no numerical packages; SymPy is approved.

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

**Epoch 2 design session: how do physical coordinates attach to a grid?**
The first concrete implementation is a Cartesian grid (`CartesianGrid` as a
concrete `IndexedSet` with coordinate geometry). Before building it, one design
question needs to be settled: how does a grid expose the physical location of
each cell center? This unblocks the `approximates` convergence-check
infrastructure (evaluate the continuous field at cell coordinates, compare) and
the SymPy-backed field interface (what coordinate symbols does the expression
use?). Once that question is answered, the Epoch 2 build is straightforward.
