# ADR-0009 — Float64 as the default precision

> **Anticipated extension.** This ADR is expected to be edited in
> place — not superseded — once the project is ready to experiment
> with mixed precision. The edit will add an explicit per-kernel
> opt-in mechanism for lower dtypes without reversing the default
> (float64 stays the default; opt-in is the narrowing).

## Context

JAX defaults to float32 on the theory that most workloads (machine
learning, graphics) benefit more from the speed and memory win than
they lose from the precision hit. Computational astrophysics
inverts those weights: orbital integrations, cosmological distance
ladders, and long time-base simulations routinely accumulate
rounding error over 10⁶–10¹² operations, and float32's ~7 decimal
digits of mantissa is inadequate under realistic dynamic ranges.
Silent precision loss in this regime typically shows up not as an
obvious crash but as physically implausible results that survive
unit tests — exactly the failure mode the replication workflow
(ADR-0007) exists to prevent.

The mechanism for safely lowering precision on specific kernels
when performance matters is its own separate design problem.
Options exist — empirical per-kernel precision certificates, runtime
invariant monitors, adjoint-based forward-error bounds — but each
carries meaningful downstream cost (reviewer cognitive load,
maintenance surface, centralized correctness risk for the
framework-based option). That design is not urgent at Epoch 0
because there are no kernels yet whose performance warrants a
downgrade.

## Decision

**Float64 is the default and only supported precision for all
kernels, accumulators, and public API surfaces at this stage.**

Concretely:

- The package enables JAX's 64-bit mode at import time via
  `jax.config.update("jax_enable_x64", True)`.
- Public kernel signatures do not expose a `dtype=` parameter.
- No global precision flag exists; there is no supported way for a
  user or caller to switch the engine to float32 ambient precision.
- Downgrading specific kernels to float32 or lower is explicitly
  out of scope at this stage. The opt-in mechanism will be added
  as an in-place amendment to this ADR (see *Anticipated
  amendment* below) once concrete performance pressure exists and
  at least one candidate kernel is identified.

## Consequences

- **Positive:** precision-loss bugs cannot be introduced by a
  global-flag mistake. Every result the engine produces has a
  uniform, documentable precision floor. Replication targets
  (ADR-0007) can be compared against published values without
  needing to reason about precision differences between runs.
- **Negative:** float64 roughly doubles memory bandwidth and
  footprint versus float32, and many GPU tensor cores are tuned
  for float32 / bfloat16 / float16 and run float64 substantially
  slower. For Epoch 0–1 scaffolding this does not matter; it will
  matter later and is the trigger for reopening the precision
  question.
- **Neutral:** choosing float64 at import time means the package
  must be imported before any other JAX-using code that might
  have already set the default — standard JAX guidance, but worth
  flagging for users who pre-import JAX in their own initialization
  code.

## Alternatives considered

**Float32 default, opt into float64 per kernel.** Matches JAX's
native default and maximizes GPU throughput. Rejected because the
default failure mode — a kernel that should have declared float64
but didn't — is silent physical wrongness, which is the worst
possible failure mode for a project organized around replicating
published results. The correct default, when correctness and
performance conflict, is the one whose failure mode is the less
dangerous of the two.

**Ambient precision flag (global or context-managed).** Lets users
select precision per run without changing kernel code. Rejected
because it trades one form of "how did this bug get here" —
per-call dtype mistakes — for a worse one: global state that
changes numerical behavior without showing up in any function
signature. The future amendment will introduce per-kernel opt-in
via an explicit `dtype=` parameter, but never via ambient state.

**Mixed-precision from day one via one of the validation
frameworks (empirical certificates, invariant monitors, adjoint
bounds).** Rejected not on effort grounds but on scope: the
project has no performance-critical kernels yet, so the framework
would validate nothing. Premature adoption would commit the
project to a specific validation style before we know which one
fits the kernels we actually build. Recorded as a deliberate
deferral, not an omission.

## Anticipated amendment

When performance pressure on a specific kernel makes a
lower-precision path worth investigating, this ADR will be amended
in place to cover:

- the mechanism for declaring a kernel's validated safe dtype
  (opt-in via `dtype=` on the public signature, a precision
  registry, or similar);
- the evidence required to bless a downgrade (empirical
  certificate, invariant monitor, a priori bound, or a
  combination);
- how downgraded kernels surface their precision contract to
  callers and to the replication harness (ADR-0007).

The three candidate validation mechanisms — per-kernel empirical
certificates, runtime invariant monitors gated behind a
`validate=` flag, and adjoint-based forward-error bounds — have
been discussed and should be re-evaluated, not re-derived, when
the amendment is written.
