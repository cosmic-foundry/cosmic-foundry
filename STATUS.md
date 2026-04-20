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
- Coordinate structure: `Chart(Function)` — diffeomorphism φ: U → V; `Atlas(IndexedFamily)`
  — charts covering M; `IdentityChart` — φ(p) = p; `SingleChartAtlas` — one global chart.
  `SmoothManifold.atlas` is an abstract property: the atlas is constitutive of the smooth
  structure. `EuclideanSpace` and `MinkowskiSpace` each provide a `SingleChartAtlas`.
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
concrete `IndexedSet` with coordinate geometry). The chart formalism is now in
place: `Chart(Function)` maps manifold points to ℝⁿ, and `EuclideanSpace` carries
a `SingleChartAtlas`. But a `CartesianGrid` is a concrete `IndexedSet`, not a
manifold — so a `Chart` cannot directly act on it. The design question is: what
object maps grid indices to physical coordinates, and how does it relate to the
chart on the ambient `EuclideanSpace`? Settling this unblocks the `approximates`
convergence-check infrastructure (evaluate the continuous field at cell
coordinates, compare) and the SymPy-backed field interface (what coordinate
symbols does the expression use?).
