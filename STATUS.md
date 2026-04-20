# Cosmic Foundry — Status

The repository is organized into four packages with a strict dependency order:

- **Foundation** (`foundation/`) — `Set`, `Function`, `IndexedSet`, `IndexedFamily`.
- **Continuous** (`continuous/`) — manifolds, fields, operators, boundary conditions.
- **Discrete** (`discrete/`) — scheme description on finite index sets; symbolic.
- **Numerical** (`computation/`) — JAX evaluates the discrete description.

`foundation/`, `continuous/`, and `discrete/` are symbolic-reasoning layers:
no floats, no numerical packages; SymPy is approved.

## Near-term design questions

**Does `ManifoldWithBoundary` belong in the hierarchy?**
`ManifoldWithBoundary` sits outside the `SmoothManifold`/atlas branch, so it has
no smooth structure and no hook for coordinate charts. Charts on a manifold with
boundary map to the closed half-space ℝⁿ₊, not to ℝⁿ, so it cannot simply
inherit the existing `Atlas` interface. The only concrete use is `Region`, which
is itself fully abstract. Options: collapse to `Region(Manifold)` directly and
defer the half-space chart question, or keep `ManifoldWithBoundary` and resolve
its relationship to the smooth/atlas hierarchy now. Either way, the current state
is carrying unresolved debt.

## Near-term work

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
