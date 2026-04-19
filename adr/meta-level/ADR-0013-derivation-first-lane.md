# ADR-0013 — Derivation-first lane for physics capabilities

## Context

Physics capabilities enter the engine by two broad paths: **adaptation**
of algorithms from an existing reference code, or **re-derivation** from
the literature. Today the repository has policy for only the first half of
that story.

**What exists.**

- [ADR-0007](ADR-0007-replication-workflow.md) defines a
  bounded-increment, verification-first workflow with per-capability specs,
  golden-data harnesses, and externally-grounded tests. It governs
  *whether* an implementation is correct once written, not *which path*
  produced it.
- [`replication/formulas.md`](../../replication/formulas.md) is a flat
  register of individual physics formulas, catching coefficient-, sign-,
  and variant-level transcription drift. Its grain is one equation per
  entry — too fine for capability-level narrative.
- [`research/06-12-licensing.md`](../../research/06-12-licensing.md) and
  [`ROADMAP.md`](../../ROADMAP.md) state a
  licensing principle:
  copyleft references (GADGET-4, RAMSES, MESA, SWIFT, PLUTO, Arepo, ...)
  are consulted *through published papers only*; any reimplementation is
  clean-room.

**What is missing.**

1. **Operational discipline for clean-room re-implementation.** The
   licensing principle is principle-only. There is no template for
   recording that a capability was clean-room derived, no agent routing
   rule forbidding opening a copyleft source tree, and no convention
   substituting for the classic firewall between paper-reader and
   code-author (which in agent-assisted development is typically a single
   agent).
2. **Capability-level home for first-principles derivations.** Several
   near-term capabilities — generalized aprox-family rate networks,
   custom stiff integrators, dynamic-NSE network grouping — are cases
   where *understanding* the formalism is the goal. Direct porting, even
   from a permissive reference, does not produce the artifact we need.
   The formula register is too fine-grained; the capability spec is the
   right level, but its current shape is engine-interface-oriented and
   does not carry step-by-step derivations or symbolic checks.
3. **A principled way to disagree with the literature.** When we believe
   published code or a published equation is wrong, we need somewhere to
   record *why* with enough rigor that the disagreement is auditable — not
   a comment in a source file and not a footnote in a spec.

This ADR closes those gaps by introducing a **derivation-first lane**
alongside the existing adapt-from-reference lane, and by naming the
three paths explicitly so that every physics-capability PR states which
path it took.

## Decision

Adopt a three-lane model for physics-capability work. Each PR touching a
physics capability declares a lane in its description; Lanes B and C
require a companion **derivation document** under a new top-level
directory `derivations/`.

### The three lanes

**Lane A — Port-and-verify.** *Default for capabilities backed by a
permissively-licensed reference code.* The reference source may be read;
algorithmic adaptation with attribution is the expected pattern. The
capability still follows ADR-0007 (spec before code, externally-grounded
test, golden fixture where applicable). No derivation document required.

**Lane B — Clean-room from paper.** *Required when the only reference
code is copyleft-licensed.* The reference source tree **must not be
opened** while the capability is being designed or implemented — not by
the human, not by the agent, not through diff previews in PR reviews of
other repositories. Authors and reviewers work from published papers and
vendor documentation. A derivation document is required; it cites papers,
not code. The agent firewall substitute (see *Clean-room firewall* below)
applies.

**Lane C — First-principles origination.** *For generalizations,
extensions, and novel work.* The derivation is the authority; published
work is consulted as a validation check, not as a blueprint. Principled
disagreements with the literature are documented in the derivation with
enough rigor that a reviewer can evaluate them.

Lane A and Lane C may coexist for a single capability when the engine
ships both a faithful reproduction and a generalization — the two get
separate PRs, separate capability specs, and in the Lane C case, a
derivation document.

### Derivation documents

Location: `derivations/DER-NNNN-<short-title>.md`. One document per
derivation. The document is a capability-level artifact; the
per-formula register in `replication/formulas.md` remains unchanged
and continues to catch coefficient-level drift. A single derivation
may generate multiple formula entries; the two are complementary.

Required sections:

1. **Physical problem.** Continuous physics statement — equations,
   domain, units, boundary conditions. No code.
2. **Discrete target.** What the engine needs to compute: the
   algorithmic result the derivation must arrive at (a stencil,
   a rate expression, an update formula, an integrator step).
3. **Derivation.** Step-by-step algebra / analysis taking the
   physical problem to the discrete target. Load-bearing steps
   carry an executable **SymPy check** (see below).
4. **External validation.** Published benchmarks, analytical
   special cases, or reference code output the derivation is
   checked against. DOI or ADS bibcode per the citation convention
   in `replication/formulas.md`.
5. **Principled disagreement.** Cases where the derivation reaches
   a conclusion that differs from a published equation or reference
   code. Each item carries the source of the disagreement, the
   reasoning, and the validation evidence supporting our version.
   If empty, state "None" and delete the subsection body.
6. **Lane and license provenance.** One of `A`, `B`, or `C` with a
   one-line justification. For Lane B, name the papers consulted
   and explicitly record that the reference code was not read.
7. **Linked capability specs.** IDs of `capabilities/`
   entries that consume this derivation.

A `derivations/README.md` will be authored in a follow-up PR alongside
the first derivation (following the ADR-0007 → `replication/README.md`
pattern). The template above is the decision artifact this ADR is
ratifying; the README refines it as experience accumulates.

### Executable symbolic checks (SymPy, not Lean)

Load-bearing algebraic steps in a derivation carry an inline SymPy
expression that, when evaluated, reduces to `True` or to a canonical
form. These run as part of the repository's standard pytest suite so a
broken derivation fails CI.

**Why SymPy, not Lean.** Lean and comparable proof assistants would
provide machine-checked proofs but add a heavy build dependency, a new
language, and a much steeper cost per derivation. SymPy is already a
baseline dependency (ADR-0007 Amendments cite it for stencil derivation)
and covers the realistic failure modes — sign errors, missed terms,
wrong variant picked from several. Lean remains available for any
single derivation that warrants it; it is not the default.

**When SymPy cannot express the check.** Some derivations (e.g.
numerical integration of a BVP, asymptotic expansions) resist a compact
symbolic check. In those cases the derivation records the fact
explicitly and falls back to a numerical-validation block — the
derivation is not downgraded, and the step is not silently dropped.

### Clean-room firewall (Lane B)

Classic clean-room discipline separates the person reading the
reference from the person writing the code. In agent-assisted
development a single agent typically does both. The substitute:

- The agent must not open the reference code (not `git clone`, not
  `gh repo view`, not a vendored mirror, not a cached search result
  whose preview contains source). Papers and vendor documentation
  only.
- The derivation document records *which papers* were read and states
  that reference code was not consulted.
- The reviewer spot-checks: if the derivation's step order, variable
  naming, or algebraic structure tracks a specific reference
  implementation suspiciously closely, that is a finding.

If a Lane B capability later needs verification against a copyleft
reference's *output* (not source), that is permitted — running the
reference code and recording its numerical results is an
observational act, not a reading of source.

### PR discipline

- PR description states the lane in the first paragraph.
- Lane B and Lane C PRs link the derivation document.
- Capability spec's `External reference` field (ADR-0007 Amendments)
  is satisfied by the derivation for Lanes B and C; the derivation
  document subsumes this requirement.

### Scope

This ADR applies only to *physics capabilities* as defined in ADR-0007
Amendments — modules implementing a physical equation, numerical
scheme, or derived quantity. Infrastructure capabilities (dispatch,
mesh topology, I/O, field placement) are explicitly out of scope;
their grounding is structural and already governed by ADR-0007.

## Consequences

- **Positive.** The copyleft licensing principle moves from
  honor-system to enforceable-in-review. A derivation-document
  requirement gives reviewers a concrete artifact to inspect.
- **Positive.** First-principles work (aprox generalization,
  custom integrators, dynamic-NSE grouping) gains a durable home.
  The *understanding* becomes a first-class artifact, not a
  side-effect of the code.
- **Positive.** The principled-disagreement pattern legitimizes
  correcting the literature without relaxing discipline: the bar is
  higher, not lower, than matching a published number.
- **Positive.** Attribution becomes clearer. Lane A cites code;
  Lane B cites papers; Lane C cites neither as authority.
- **Negative.** Derivation authoring is slower than porting. The
  cost is contained by scoping: Lane A (permissive reference) needs
  no derivation, and the formula register continues to handle
  coefficient-level drift without escalation.
- **Negative.** The agent firewall for Lane B is behavioral, not
  technical; a careless agent could violate it. Mitigation lives in
  the AI.md routing rule and reviewer spot-checks; strict tooling
  (e.g. automated contamination detection) is deliberately not
  proposed here.
- **Neutral.** Introduces a new top-level directory `derivations/`,
  parallel to `replication/`. They do distinct jobs and reference
  each other via capability IDs.
- **Neutral.** Existing Epoch 1 capabilities are grandfathered as
  Lane A by default; no back-fill is required. Epoch 6+ microphysics
  work is the first expected heavy user.

## Alternatives considered

- **Status quo (ADR-0007 alone).** Keeps the workflow simple but
  leaves the licensing principle unimplemented and gives no home
  for capability-level derivations. The stated pattern
  ("re-derive, don't port") has no artifact to anchor to.
- **Extend `replication/formulas.md` instead of a new directory.**
  The formula register's grain is per-equation; derivations are
  per-capability and carry narrative plus symbolic checks. Forcing
  them into the same file blurs the levels and discourages the
  step-by-step structure a derivation needs.
- **Mandatory derivations for every physics capability.** Strictly
  safer but kills throughput where the payoff is low (textbook
  operators, standard second-order stencils). The lane model
  preserves the option of Lane A for genuinely routine work.
- **Lean as baseline proof assistant.** Strongest grounding but a
  heavy, unfamiliar dependency for a codebase whose algebra is
  mostly routine. SymPy covers the realistic risks today; Lean
  stays available as an opt-in per-derivation tool.
- **Automated copyleft-contamination scanning.** Token-level
  comparison between the engine and known copyleft codebases would
  enforce Lane B mechanically. Rejected now as high-cost
  infrastructure; the derivation-document discipline plus reviewer
  attention is the cheaper first defense. May be revisited if Lane
  B capabilities become frequent.

## Architecture stress-review note

Per `pr-review/architecture-checklist.md`, required for ADR PRs.

### 1. Problem Boundary

This ADR adds a discipline (derivation-first lane) and a directory
(`derivations/`) for physics capabilities. Downstream costs it
reduces: (a) legal / compliance risk from inadvertent copyleft
contamination, (b) reviewer load when provenance of a physics
implementation is unclear, (c) reversibility — derivations make
past decisions auditable and changeable, (d) correctness of
generalizations, which require understanding, not just a correct
port.

Terms defined above: *physics capability* (ADR-0007 Amendments),
*port*, *clean-room*, *origination*, *derivation*, *reference code*.

Out of scope: infrastructure capabilities; the choice of proof
assistant technology; automated contamination detection; changes
to `replication/formulas.md`'s entry criteria.

### 2. Tiling Tree

**Split 1 — by provenance of the algorithm.**

- (a) permissive-licensed reference code exists → Lane A default.
- (b) copyleft-licensed reference code is the only reference → Lane B
  mandatory.
- (c) no reference code, or we intend to generalize / originate →
  Lane C.
- (d) reference is only a paper (no code) → Lane C in practice;
  trivially satisfies Lane B's clean-room condition (nothing to
  avoid reading).

Branches (a)–(d) are chosen per-PR and are mutually exclusive for a
given capability version.

**Split 2 — by grounding artifact level.**

- (e) per-formula (`replication/formulas.md` entry).
- (f) per-capability (derivation document under `derivations/`).
- (g) per-problem (golden fixture under `replication/targets/...`).

These are complementary, not exclusive. A Lane B/C capability can
have all three; a Lane A capability often has (e) and (g) but no
(f).

**Split 3 — by verification mechanism inside a derivation.**

- (h) SymPy-executable symbolic check (default baseline).
- (i) analytical special-case value (e.g. ∇²(x²+y²+z²) = 6).
- (j) published benchmark number with citation.
- (k) numerical reference-code output (Lane A or output-only Lane B).
- (l) proof-assistant formalization (opt-in, not required).

Lower-complexity branch considered and rejected: "no new directory
— just add an AI.md paragraph telling agents to re-derive when
appropriate." Rejected because it leaves no durable artifact; the
SymPy check has nowhere to live, principled disagreements have
nowhere to be recorded, and each future capability re-litigates the
lane choice from scratch. The cost saved is one directory; the
downstream cost is continuously escalating.

Unexplored leaf, marked: tooling to detect token-level copyleft
contamination. Not proposed here.

### 3. Concept Ownership Table

| Concept | Owns | Does not own |
|---|---|---|
| Lane A (port-and-verify) | adaptation from permissive reference, attribution | original derivation |
| Lane B (clean-room) | paper-only implementation, firewall discipline | direct code adaptation |
| Lane C (origination) | first-principles derivation, principled-disagreement discipline | mere transcription |
| Derivation document | per-capability narrative, symbolic checks, principled-disagreement record | per-formula coefficient entries |
| `replication/formulas.md` | per-formula coefficient / sign / variant grounding | capability-level narrative |
| Capability spec (ADR-0007) | engine-interface contract, external-grounding pointer | step-by-step derivation |
| AI.md routing rule | lane-proposal heuristic for agents | lane selection (user decides) |

No fuzzy cells.

### 4. Real Workflow Stress Test

**Workflow 1 — "Give me that new aprox-family extension pynucastro
just added."** pynucastro is permissively licensed, so Lane A is the
default. But the user's framing — "extend the family, ours may need
to generalize further" — signals intent to originate. The agent
proposes Lane C. User confirms. The PR includes
`derivations/DER-000N-aprox-family-generalization.md` with the
physical problem, discrete target, step-by-step derivation of the
rate formalism, SymPy checks of the algebraic identities, and a
(likely empty at first) principled-disagreement section. The Lane C
capability spec points to the derivation; the formula register
gains one entry per rate coefficient as they materialize.

**Workflow 2 — "Replicate the MESA atmospheric-boundary scheme."**
MESA is LGPL. Lane B is mandatory. The agent must not open the MESA
source; MESA papers and documentation are the inputs. The derivation
document records the papers consulted and states that MESA source
was not read. The capability spec's `External reference` field
points to the derivation. The reviewer spot-checks that the
derivation's structure does not track MESA's implementation
suspiciously closely.

Both workflows express naturally in the ADR's vocabulary. No
missing concept surfaced.

### 5. Normalization Trace

```
author writes derivation (derivations/DER-NNNN-*.md)
  -> SymPy checks execute in pytest (CI-gated)
  -> capability spec (capabilities/CNNNN-*.md) links the derivation
  -> engine implementation lands with capability spec + derivation + golden tests
  -> formula register gains per-formula entries where coefficients are load-bearing
```

Nothing runs at engine runtime as a result of this ADR. The
materialization point is CI-time: a broken SymPy check blocks
merge.

### 6. Ordering, Visibility, And Fences

No runtime ordering. The SymPy check is the sole automated fence,
and it operates at test-collection time. Principled-disagreement
sections are advisory for human reviewers; no machine check. The
clean-room firewall is enforced by discipline and review, not by
tooling.

### 7. Backend / Lowering Trace

No backend layer. This ADR is a documentation and process
decision.

### 8. Alternative Failure Pass

- **Where does this abstraction collapse?** When a derivation's
  core step resists compact symbolic expression (BVP numerics,
  asymptotic expansions). Mitigation: the ADR explicitly permits
  falling back to a numerical-validation block; the derivation is
  not forced into ceremony.
- **Which concept is secretly doing another concept's job?** Risk:
  the derivation document drifts into being the capability spec.
  Prevented by keeping the spec as the engine-interface artifact
  that *links* the derivation, not inlines it.
- **Ordering dependency missed?** None; derivation artifacts are
  atemporal.
- **What would change semantics if an implicit boundary moved?**
  The lane label is advisory metadata, but a mislabel (Lane A
  where Lane B applies) is a licensing risk. Mitigation: AI.md
  forces explicit agent-proposed, user-confirmed lane selection;
  reviewers check the lane label against the reference license.
- **What would an author need that this ADR does not specify?**
  A decision procedure for "when is SymPy ceremony?" is
  deliberately left to experience; the first few derivations will
  shape the norm.
- **Simpler option rejected on downstream-cost grounds?** The
  "AI.md paragraph only" option — rejected because the lack of a
  durable artifact re-litigates the decision on every future PR,
  which is a recurring reviewer-load cost, not a one-time
  authoring cost.

### 9. Decision Delta

**Ready with named risks.**

- **R1 — SymPy-inexpressible derivations.** Mitigated by explicit
  fallback clause; watch the first 2–3 derivations to confirm.
- **R2 — Agent mislabels a lane.** Mitigated by AI.md routing
  rule and reviewer spot-check against the license-class
  inventory in `research/06-12-licensing.md`.
- **R3 — Scope creep onto infrastructure.** Mitigated by the
  explicit scope clause; revisit if experience argues otherwise.
