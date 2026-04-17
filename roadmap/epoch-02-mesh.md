# Epoch 2 — Mesh and AMR

> Part of the [Cosmic Foundry roadmap](index.md).

## Design prerequisites

Two design decisions must be made **before** the task-graph driver is
implemented — they cannot be resolved mid-epoch without restructuring
work already in flight.

### Field name → Dispatch input binding

Epoch 1 `Op` metadata declares `reads=("phi",)` and `writes=("laplacian_phi",)`,
but `Dispatch.inputs` accepts positional raw arrays with no enforcement that
input[0] corresponds to the "phi" field.  The Epoch 2 task-graph driver must
answer: *given that a Dispatch reads field "phi", where in the field registry
is phi, which local segment belongs to this rank, and which input position does
it bind to?*

Design the binding layer first.  Options include a `fields: dict[str, Field]`
argument on `Dispatch`, a separate `FieldBinding` object, or a driver-owned
field-resolution step that extracts local payloads and constructs the
positional inputs before calling `execute`.  Record the decision as an
amendment to ADR-0010 before opening the task-graph PR.

### Halo fill fence

`Field.covers(required)` validates that coverage *exists*, but there is no
mechanism for the driver to *fill* missing halos.  The Epoch 1 multi-rank
worker sidestepped this by pre-slicing overlapping arrays from a locally
available full domain; production domain decomposition cannot do that.

Before implementing ghost-cell exchange, define the fence primitive: something
like `HaloFillFence(field, region, access_pattern)` that the driver inserts
before any `Dispatch` whose required footprint extends beyond a segment's
interior.  This is where `jax.distributed` collectives live.  The
`Field`/`Placement`/`Region` metadata model has all the information needed;
the orchestration layer does not yet exist.

### Global reduction primitive for simulation diagnostics

Physics simulations need timestep-by-timestep records of global derived
quantities — total mass, momentum, energy, etc. — both for production
monitoring and as a basis for conservation-law tests.  This is the pattern
Castro uses: a `.diag` file written every step that records these integrals,
useful for spotting non-physical behavior mid-run and for replicating against
reference code output.

Epoch 2 mesh operators must expose a primitive — something like
`global_sum(field, region)` — that the future simulation driver can call to
accumulate domain-wide integrals across ranks.  Design this interface
alongside the domain-decomposition work so that physics modules added in later
epochs can register diagnostics against a stable contract.

**Conservation tests must document their validity conditions.**  Not all
global quantities are conserved by all schemes under all boundary conditions:

- *Conservative scheme + periodic or reflecting BCs:* mass, momentum, and
  energy are conserved to machine precision.  Asserting exact conservation
  is a valid test.
- *Outflow BCs:* quantities leak through the boundary by design; the domain
  is not a closed system.  Simply asserting that global Q is constant will
  always fail and should not be attempted.  Instead, use the stronger
  **boundary-flux balance test**: for a conservative finite-volume scheme the
  discrete conservation law is Q(tₙ₊₁) = Q(tₙ) − Σ(boundary face flux × Δt),
  and this identity holds to machine precision regardless of BC type.  Record
  the accumulated boundary flux as a diagnostic alongside Q and assert the
  balance.  This is a more demanding check than the closed-system test because
  it verifies that the numerical fluxes are self-consistent with the
  cell-average updates — a fundamental property of the conservative form that
  the closed-system test cannot distinguish from a coincidentally correct
  answer.  It requires that boundary face fluxes be accessible as a diagnostic
  output; the driver interface should expose this from the start rather than
  retrofitting it later.
- *Intentionally dissipative schemes* (PPM with limiting, entropy fixes,
  artificial viscosity): energy dissipation is a feature.  The diagnostic
  tracks dissipation rate; tests should assert the *form* of dissipation
  (e.g. entropy non-decreasing) rather than total energy conservation.

Tests that assert conservation must state in their docstring which of these
conditions hold.  A conservation test without this documentation is incomplete
and should be treated as a review finding.

---

## What this epoch delivers

Bring up the mesh hierarchy the physics modules will live on:

- Uniform structured grid with ghost-cell exchange and domain
  decomposition.
- Block-structured AMR hierarchy expressed as JAX pytrees of blocks
  with cell / face / edge / node centering, subcycling in time, and
  refinement-flux correction.
- Task-graph driver for asynchronous dependency scheduling across
  ranks and devices.
- Plotfile writer and yt-compatible metadata.
- Zarr v3 writer with an OME-Zarr-style multiscale pyramid
  convention emitted alongside the plotfile, so every AMR output
  is simultaneously HPC-analysis-ready (HDF5 / plotfile) and
  browser-streamable (Zarr). Pinned by ADR-0006.

**Exit criterion:** a second-order advection test converges at
design order on AMR, runs identically on CPU and GPU, produces
rank-invariant output, and writes a matching Zarr pyramid that
the Epoch 3 viewer MVP consumes without hand editing.
