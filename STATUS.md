# Cosmic Foundry — Status

The repository is in a foundation-building phase. We have cleared away
wrong architectural assumptions (the old Op/Region/Policy framing, bloated
Function model, mixed concerns in mesh/) and rebuilt on a clean slate:

- `theory/` — pure mathematical ABCs (complete)
- `geometry/` — Domain and manifold concretizations (complete)
- `computation/` — Array and Extent data containers (complete)

There is currently no planned work in this repository. The next steps will
build the computation layer (stencils, reductions, field operations, spatial
sweeps) from first principles, informed by the architectural basis in
[`ARCHITECTURE.md`](ARCHITECTURE.md).

For the long-horizon development sequence, see [`ROADMAP.md`](ROADMAP.md).
