# ADR-0012 — Global Reduction Primitive

## Context

Physics simulations need timestep-by-timestep records of domain-wide
integrals — total mass, momentum, energy, and similar conservation-law
quantities.  These serve two distinct purposes:

1. **Production monitoring** — detecting unphysical behavior mid-run
   (e.g. negative density, sudden energy spike) without post-processing.
2. **Verification testing** — conservation-law tests that replicate against
   reference code output and enforce stated validity conditions.

Many production astrophysics codes write a small text file every step
recording these scalars — a lightweight, human-readable record that is easy
to monitor in a terminal and diff against reference output.

The `Field`/`Region`/`Dispatch` model provides the computational substrate,
but there is no current primitive for summing a field quantity across all
ranks and returning a single scalar.  `jax.lax.psum` provides the collective,
but it operates on in-scope JAX values — the driver needs a higher-level
contract that names the quantity, records its validity conditions, and exposes
boundary fluxes alongside volume integrals so that outflow-BC conservation
tests can be written correctly.

### Validity conditions

Not every simulation conserves every quantity.  The diagnostic contract must
distinguish three cases so that callers are not tempted to write tests that
silently pass for wrong reasons:

- **Closed system** (conservative scheme + periodic or reflecting BCs):
  mass, momentum, and energy are conserved to machine precision.
  `assert global_Q(t+dt) == global_Q(t)` is a valid test.
- **Outflow BCs**: quantities leave the domain through boundary faces.
  `global_Q` is not constant.  The correct check is the
  *boundary-flux balance*:
  `global_Q(t+dt) == global_Q(t) − boundary_flux_integral(t→t+dt)`.
  This requires that boundary face fluxes be exposed as diagnostics.
- **Dissipative schemes** (PPM with limiting, entropy fixes, artificial
  viscosity): energy dissipation is intentional.  Tests should assert the
  *form* of dissipation (entropy non-decreasing) rather than exact
  conservation.

A diagnostic that does not document which case applies is incomplete.

## Decision

Introduce a `DiagnosticReducer` protocol and a `DiagnosticRecord` container.
The driver calls reducers each step and collects the results into a record
written to a `.diag`-style output file.

### Concepts

**`DiagnosticReducer`** is a protocol (structural, like `OpLike`):

```python
class DiagnosticReducer(Protocol):
    name: str               # e.g. "total_mass", "total_energy"
    includes_boundary_flux: bool

    def reduce(
        self,
        fields: dict[str, Field],
        region: Region,
        rank: int,
        n_ranks: int,
    ) -> jax.Array:
        """Compute the global scalar value for the current timestep.

        Returns a 0-d JAX array (not a Python float) so that all
        registered reducers can be collected inside a single
        ``jax.jit`` call, enabling XLA to fuse their ``psum``
        collectives into one cross-rank round-trip.
        """
```

`reduce` returns a 0-d `jax.Array`.  Implementations call
`jax.lax.psum` (or equivalent) for the cross-rank sum and return
the result without materializing a Python `float`.  The driver
extracts Python floats after the full collection step completes.

**Reduction fusion:** XLA has an all-reduce fusion pass that
combines multiple `psum` calls within the same JIT-compiled
computation into a single collective over a vector.  This is the
key reason `reduce` returns a `jax.Array` rather than a `float`:
if the driver wraps `collect_diagnostics` in `jax.jit`, XLA sees
all N reductions together and fuses them into one round-trip at the
cost of N scalars of bandwidth — essentially free for any realistic
N of registered reducers.  Returning `float` inside each reducer
would force N independent device-to-host transfers and break the
JIT boundary, defeating fusion.

**Runtime health checks:** value-dependent error checks follow the same
rule. A check for non-finite values, negative density or pressure,
CFL violation, or another simulation health condition should reduce to a
0-d JAX array or a small JAX vector. The driver materializes those results
only at an explicit diagnostic or health-check fence, then raises any
host-visible warning or exception. Structural checks that inspect only
metadata — extents, ranks, field names, and dependency ordering — remain
ordinary Python validation before launch.

**`DiagnosticRecord`** is a named tuple row emitted each step:

```python
@dataclass(frozen=True)
class DiagnosticRecord:
    step: int
    time: float
    values: dict[str, float]   # reducer.name → scalar value
```

**`DiagnosticSink`** writes records.  The baseline implementation appends
a tab-separated row to a `.diag` text file (one column per registered
reducer).  A null sink for unit tests discards records without I/O.

**Driver usage pattern:**

```python
# Registration (once, before the time loop):
driver.register_diagnostic(TotalMassReducer())
driver.register_diagnostic(TotalEnergyReducer())

# Each step (after updating fields, before writing output):
record = driver.collect_diagnostics(fields, region, step, time)
sink.write(record)
```

`collect_diagnostics` calls each registered `DiagnosticReducer.reduce` in
order and returns one `DiagnosticRecord`.  The driver is responsible for
calling `collect_diagnostics` at a consistent point in the step — after
the update but before the next boundary exchange, so that the recorded
values correspond to a single coherent state.

### Primitive-level helper

For the common case of summing a cell-averaged quantity over a region,
provide a free function:

```python
def global_sum(field: Field, region: Region, rank: int) -> jax.Array:
    """Sum field values over the local segment interior and reduce across ranks.

    Returns a 0-d JAX array so the result participates in XLA fusion
    when called from within a jax.jit-compiled collection step.
    """
```

`global_sum` sums `field.local_segments(rank)` restricted to
`region.extent`, then calls `jax.lax.psum` across all ranks.  This is the
building block most `DiagnosticReducer.reduce` implementations will use; it
is not the entire interface.

### Conservation test documentation requirement

Any test that invokes a `DiagnosticReducer` or asserts a conservation law
**must** document in its docstring which validity condition applies, following
this pattern:

```python
def test_mass_conservation_closed_domain() -> None:
    """Total mass is conserved to machine precision.

    Validity: conservative FV scheme, periodic BCs — closed system.
    Method: assert |Q(t+dt) − Q(t)| < 1e-12 * Q(t).
    """
```

Tests without this documentation are incomplete and should be treated as a
review finding (see `roadmap/object-level/epoch-02-mesh.md`).

### Boundary-flux balance test pattern

For outflow BCs, the boundary-flux balance test requires recording the face
flux at boundary cells as a diagnostic:

```python
driver.register_diagnostic(BoundaryFluxReducer(axis=0, face="hi"))
```

A `BoundaryFluxReducer` sums the outward-normal flux on one boundary face.
The test then asserts:

```python
assert abs(Q_new - Q_old + boundary_flux * dt) < tol
```

This is the more demanding check: it verifies that numerical fluxes are
self-consistent with the cell-average updates, not merely that a scalar
appears conserved coincidentally.  Exposing boundary fluxes as first-class
driver outputs is required from the start; retrofitting it after the driver
is built has historically been costly.

### Relationship to AMR flux correction

`BoundaryFluxReducer` and AMR flux correction (refluxing) share the same
data dependency: face fluxes must be first-class driver outputs rather than
temporaries discarded after the cell-average update.

The operations diverge after that.  `BoundaryFluxReducer.reduce` returns a
0-d `jax.Array` like every other `DiagnosticReducer`; the driver converts it
to a Python `float` only after the batched diagnostic collection step
completes and it is building a host-visible `DiagnosticRecord`.  AMR flux
correction returns an *array* — a correction to coarse cell averages at
coarse-fine interfaces, needed to make block-structured AMR globally
conservative.  The `DiagnosticReducer` protocol, which produces scalar
diagnostic values, is too narrow to express flux correction; that is a
separate interface (a flux register) to be designed in the AMR sub-epoch.

The implication for this ADR: the face-flux accessibility requirement is
motivated by *two* downstream uses, not one.  Designing the driver to expose
face fluxes from the start satisfies both the outflow-BC diagnostic use case
and the future refluxing requirement without interface changes.

## Consequences

- **Positive:** The same diagnostic write path serves both production
  monitoring and conservation tests — there is no separate reimplementation
  of domain integrals for tests.
- **Positive:** The docstring requirement on conservation tests makes validity
  conditions explicit and reviewable at the point where the assertion is made,
  rather than on the reducer (which is indifferent to the simulation setup).
- **Positive:** The `.diag` text format is a common convention in
  astrophysics codes; it is human-readable and diff-friendly.
- **Positive:** `DiagnosticSink` is swappable; tests use a null sink; the
  real sink writes to disk without modifying test code.
- **Neutral:** N registered reducers produce N `psum` calls per step, but
  XLA's all-reduce fusion combines them into one collective when
  `collect_diagnostics` is JIT-compiled — collective overhead scales with
  bandwidth (N scalars), not with latency (N round-trips).  Reducers should
  still be cheap to evaluate locally; the cross-rank cost is fused away.
- **Neutral:** `DiagnosticReducer` is a protocol, not a base class.
  Implementations may be functions, lambdas, or class instances; the
  driver does not care which.

## Alternatives considered

**`jax.lax.psum` called directly from physics kernels** — Embed the
reduction inside the Op function.  This makes the reduction invisible to
the driver (cannot be disabled, reordered, or batched) and makes Ops
stateful.  Ruled out; violates the stateless-Op contract from ADR-0010.

**Single `global_sum` free function, no protocol** — Expose only the
primitive helper and require callers to manage their own records.  Simpler
initially, but leaves the registration / write-every-step pattern
unspecified, so each physics module would invent its own diagnostic loop,
leading to duplicated and inconsistent integration logic.  Ruled out.

**Structured array output (HDF5 per-step diagnostic group)** — Write
diagnostics as HDF5 datasets rather than a text file.  More structured,
queryable.  But for the primary use case (monitoring a run in a terminal
and diffing against reference output) a tab-separated text file is faster
to read and easier to grep.  HDF5 diagnostic output can be added as a
second `DiagnosticSink` implementation later without changing the protocol.
Deferred.

**Pull-style diagnostics (driver queries reducers on demand)** — Register
reducers and let post-processing tools query them after the run.  Requires
checkpointing intermediate states or replaying the simulation.  Incompatible
with the monitoring use case.  Ruled out.
