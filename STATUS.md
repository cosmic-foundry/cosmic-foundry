# Cosmic Foundry — Status

The repository is organized into four packages with a strict dependency order:

- **Foundation** (`foundation/`) — `Set`, `Function`, `IndexedSet`, `IndexedFamily`.
- **Continuous** (`continuous/`) — manifolds, fields, operators, boundary conditions.
- **Discrete** (`discrete/`) — scheme description on finite index sets; symbolic.
- **Numerical** (`computation/`) — JAX evaluates the discrete description.

`foundation/`, `continuous/`, and `discrete/` are symbolic-reasoning layers:
no floats, no numerical packages; SymPy is approved.

## Near-term work

**Resolve the BoundaryCondition hierarchy.**
One open question remains:

1. ~~`BoundaryCondition(Function[D, C])`~~ — resolved. `Constraint(ABC)` is
   the new root with abstract `support: Manifold`; `BoundaryCondition` inherits
   it. The `Function` base and `[D, C]` type parameters are gone.

2. `LocalBoundaryCondition.constraint` is a `Field`, but the manifold it is
   defined on is unspecified. The constraint lives on the boundary face; the
   continuous layer has no formal type for a boundary face.

**M2.5 design session: mathematical narrative documentation.**
What does the first notebook look like, and how does it hook into CI? Concrete
questions to settle: (1) which concept is the right entry point — `Set` and
`Function`, or the manifold hierarchy? (2) what makes a notebook "tested" — does
it run SymPy derivations that assert results, does it instantiate the ABCs, or
both? (3) how do we structure the `docs/` tree so notebooks accumulate without
becoming a maze? The goal is a format that can be repeated for every new concept
added to `continuous/` and `discrete/` as the epochs proceed.

**Epoch 2 design session: how do physical coordinates attach to a grid?**
The first concrete implementation is a Cartesian grid (`CartesianGrid` as a
concrete `IndexedSet` with coordinate geometry). The chart formalism is in
place: `Chart(Function)` maps manifold points to ℝⁿ, and `EuclideanSpace` carries
a `SingleChartAtlas`. But a `CartesianGrid` is a concrete `IndexedSet`, not a
manifold — so a `Chart` cannot directly act on it. The design question is: what
object maps grid indices to physical coordinates, and how does it relate to the
chart on the ambient `EuclideanSpace`? Settling this unblocks the `approximates`
convergence-check infrastructure (evaluate the continuous field at cell
coordinates, compare) and the SymPy-backed field interface (what coordinate
symbols does the expression use?).
