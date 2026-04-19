# ADR-0008 — Numerical-transcription discipline (placeholder)

> **Stub.** This ADR reserves number 0008 and records the
> problem framing so future work does not re-derive it from
> scratch. The decision itself is intentionally deferred: it
> should not be forced before Epoch 1 has shaped the kernel
> interface (ADR-0002) and before any real transcription work is
> in flight. Final text lands before Epoch 7 (Microphysics)
> begins, where the need becomes concrete.

## Context

Porting files like `aprox_rates.H` from AMReX-Astro/Microphysics —
dense collections of analytic reaction-rate formulas for dozens of
nuclear reactions, each with carefully tuned numerical coefficients
— is the class of work Cosmic Foundry is most likely to make silent
mistakes on. Characteristics:

- **Not splittable into ~100-LOC logical increments without losing
  coherence.** A single reaction rate with its low-T / high-T
  branches and interpolation tables is an atomic unit of scientific
  content; splitting it across commits makes no sense.
- **Very high defect sensitivity.** A transcription error in the
  fifth significant figure of a coefficient does not cause a
  compile / test failure — it produces a quietly-wrong rate that
  systematically biases downstream physics.
- **Low algorithmic novelty.** The work is ~90% transcription,
  ~10% adapting syntax. The reviewer's job is digit-by-digit
  correctness, not architectural judgment.

Two process rules already in force interact with this:

- **ADR-0005** specifies a ~100-LOC-per-code-commit guideline.
  Designed for architectural and procedural reviewability — the
  mechanism it relies on (a reviewer reading a small diff to
  understand the change) does not help catch transcription errors.
- **ADR-0007** specifies a verification-first, bounded-increment
  replication workflow. The real defense for transcription is here:
  golden-data tests against the reference code on a defined input
  grid catch quietly-wrong rates empirically, which no amount of
  code review can guarantee.

The tension is that ADR-0005's LOC guideline is ill-fitted to
transcription work, while ADR-0007's verification-first discipline
is exactly what transcription needs. The two rules were designed
for different failure modes.

## Candidate approaches (not yet chosen)

Captured from session discussion; each is a starting point for
the final decision, not a commitment.

1. **Transcription-PR carve-out to ADR-0005.** Define a commit
   category that is not bound by the ~100-LOC guideline provided
   it satisfies specific conditions: every numerical constant
   carries a source-file + line + paper citation; every ported
   function has a passing golden-data test at merge time; the
   diff does not mix transcription with architectural change.
2. **Lean on ADR-0007 alone.** Document that transcription PRs
   are governed by the verification-first discipline; ADR-0005's
   LOC rule is inapplicable to them by virtue of the "small
   overshoots" clause covering "logical change" atomicity. No new
   rule, just a clarification of scope.
3. **Write ADR-0008 (this slot) as a full discipline extending
   ADR-0007.** Formalize transcription-PR requirements in one
   place: source-provenance comments, golden-data coverage
   thresholds, per-file merge gates, and an explicit statement
   that LOC caps do not apply. Most durable; heaviest to draft.
4. **Symbolic-first transcription.** Per ADR-0001's symbolic-
   codegen bet, where a rate admits a closed-form SymPy
   expression, author it as a `reaction → SymPy expression`
   dictionary entry. SymPy emits the JAX kernel. Reviewer surface
   per rate collapses to one line; autodiff becomes available
   for free. Does not cover every case (piecewise branches,
   interpolation tables, empirical fits), but dramatically
   improves signal-to-noise for the common case. This is a
   partial-solution component that composes with any of
   approaches 1-3.

## Decision

**Deferred.** This ADR reserves the number, records the problem
framing, and points to ADR-0007 as the operative discipline for
any numerical transcription that happens before the final text
lands. The decision must be made before Epoch 7 (Microphysics)
begins — see `ROADMAP.md` §Crossroads / Open Decisions for
the crossroad.

## Consequences

- **Positive.** Future sessions picking up microphysics work
  start from a captured analysis rather than re-deriving it.
  ADR-0007's existing discipline covers the interim — any
  transcription PR landing before this ADR is finalized must
  satisfy ADR-0007's golden-data verification bar.
- **Negative.** The open question remains visible in the ADR
  index, which slightly increases the cognitive load of
  onboarding.
- **Neutral.** The four candidate approaches are not ranked by
  this stub; ranking waits for the data a real Epoch-6 PR will
  provide.

## Alternatives considered (for the stub vs no-stub question)

- **Roadmap crossroad only, no ADR.** Minimal but loses the
  problem-framing analysis; future work would re-derive the
  tension and the candidate approaches.
- **Full ADR written now.** Premature: no transcription work is
  in flight; the kernel interface (ADR-0002) is not yet shaped;
  the SymPy-first path's viability cannot be assessed without a
  real microphysics workload to pressure-test it.
- **Note in AI.md.** Too implementation-specific for the
  quick-reference layer.

## Cross-references

- ADR-0001 (Python-only + runtime codegen) — constrains the
  transcription target (Python + JAX / SymPy, not C++).
- ADR-0005 (branch / PR / commit-size discipline) — provides
  the LOC guideline this ADR will partially relax.
- ADR-0007 (replication workflow) — provides the verification-
  first discipline this ADR will extend.
- [`ROADMAP.md`](../../ROADMAP.md) §Crossroads /
  Open Decisions — records the
  crossroad for discoverability.
- [`ROADMAP.md`](../../ROADMAP.md)
  — the epoch where this decision becomes concrete.
