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
Three open questions following the deletion of `ManifoldWithBoundary` and `Region`:

1. `BoundaryCondition(Function[D, C])` — a BC is a constraint on a field, not a
   function mapping inputs to outputs. `__call__` is left fully abstract with no
   typed signature at the continuous layer; concrete implementations are deferred
   to `computation/`. The `Function` base is wrong. What should `BoundaryCondition`
   inherit from instead?

2. `LocalBoundaryCondition.constraint` is a `Field` — but a field defined on what
   manifold? Previously the implicit answer was "the boundary face," typed as
   `ManifoldWithBoundary`. That type is gone. The constraint's domain is now
   unspecified.

3. `NonLocalBoundaryCondition` is a bare marker with no content. Is it justified
   as a signal that "this BC is not of the Robin form," or is it premature until
   a concrete non-local BC (e.g. periodic) exists in the hierarchy?

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
