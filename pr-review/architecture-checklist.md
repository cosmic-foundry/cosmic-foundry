# Architecture stress-review checklist

Use this checklist for architectural changes, roadmap decisions, and PRs that introduce
or reshape architectural abstractions. The goal is to force the design
through the same adversarial questions a domain expert would ask before
human review.

This checklist complements `checklist.md`. The regular PR checklist
still catches repository discipline and historical failure modes; this
one tests whether a proposed abstraction has coherent ownership,
usable API shape, and defensible long-term boundaries.

## When to run

Run this checklist when a PR does any of the following:

- adds or amends an architectural claim in `ARCHITECTURE.md`;
- changes a roadmap epoch's architectural contract;
- introduces a named abstraction, protocol, base class, dispatch layer,
  task-graph concept, backend interface, storage model, or public API
  boundary;
- compares architectural options; or
- changes how physics authors, driver authors, or backend authors are
  expected to structure code.

If in doubt, run it. Architecture review is cheaper before the design
becomes vocabulary.

## Required artifact

The reviewer must produce an architecture stress-review note with these
sections. For a PR review, include the note under the normal review
report's relevant severity section or in the checklist walkthrough. For
an author self-review, include the note in the PR description before
requesting human review.

### 1. Problem Boundary

State the problem the abstraction solves in one paragraph. Define every
term that could be overloaded in this domain. Then state what is
deliberately out of scope.

Check:

- Does the problem statement name the downstream cost being reduced
  (reviewer load, operational complexity, reversibility, correctness,
  safety, blast radius), not just authoring effort?
- Are words like kernel, domain, region, stage, sweep, task, policy,
  schedule, driver, backend, field, and reduction defined concretely
  before being used as design categories?
- Is there a clear "does not own" boundary for the proposed
  abstraction?

### 2. Tiling Tree

Build a small tiling tree of the solution space before endorsing the
chosen design. The method is adapted from the "tiling tree" /
morphological-analysis idea described in
[The Tiling Tree Method](https://engineeringx.substack.com/p/the-tiling-tree-method):
split the possible solution space into subsets that are intended to be
mutually exclusive and collectively exhaustive, define each split
precisely, then recurse on the branches that matter.

For this repository, a useful tiling tree is not a long brainstorm. It
is a compact map of the design space that makes hidden alternatives and
missing leaves visible.

Required:

- At least two independent splits of the design space.
- For each split, state why the branches do not overlap and what cases
  they collectively cover.
- Include the lower-complexity branch even if it is not recommended.
- Mark unexplored leaves explicitly instead of silently dropping them.
- Evaluate the plausible leaves on downstream costs: reviewer load,
  maintenance/operations cost, reversibility, correctness/safety
  guarantees, and failure blast radius. Implementation effort is only a
  tiebreaker.

Example split prompts:

- nominal class hierarchy vs structural protocol;
- explicit user-authored fusion boundary vs transparent optimizer
  fusion only;
- field-name dependencies vs field/region/access-mode dependencies;
- driver-owned synchronization vs kernel-owned synchronization;
- one backend-specific API vs backend-neutral contract with adapters.

### 3. Concept Ownership Table

List each named concept and what it owns. Also list what it explicitly
does not own.

Use this shape:

```text
Concept       Owns                         Does not own
Op            per-element computation      iteration, scheduling
Region        iteration extent             thread organization
Policy        execution organization       physics semantics
Dispatch      lowering/fusion boundary     global synchronization
Task graph    ordering/fences/comm         per-element computation
```

The exact concepts will vary by PR. A fuzzy cell is a design defect,
not a wording issue.

### 4. Real Workflow Stress Test

Write realistic author-facing pseudocode for at least two workflows
that exercise different parts of the abstraction. At least one should
be a domain workflow, not a toy.

For kernel/driver designs, examples include:

- a CTU hydro update with an explicitly fused local algorithm;
- a stencil diagnostic with halo reads;
- a CFL or convergence reduction;
- particle deposition or gather/scatter;
- AMR synchronization or boundary exchange.

Check:

- Does the proposed API express the workflow without hidden global
  state, magical ordering, or manual reconstruction of internal
  objects?
- Does the natural code shape match the names in `ARCHITECTURE.md`?
- Does the workflow require a concept that `ARCHITECTURE.md` forgot to name?
- Does any example rely on behavior the design only permits as a
  transparent optimization?

### 5. Normalization Trace

Trace the design from author code to the internal representation that
the next layer consumes.

For each stress-test workflow, write:

```text
author code
-> normalized metadata / dependencies
-> scheduling or lowering unit
-> externally visible results
```

Check:

- Are dependencies more granular than string names when correctness
  requires it?
- Are internal temporaries hidden from outer layers unless explicitly
  materialized?
- Are in-place-looking updates versioned or otherwise disambiguated?
- Are reductions, gather/scatter, halos, and neighbor reads represented
  explicitly enough for ordering?

### 6. Ordering, Visibility, And Fences

State which dependencies order work and which boundaries prohibit
optimization across them.

Check:

- What can be reordered?
- What can be transparently fused?
- What must materialize? For JAX/GPU designs, pay particular
  attention to device-to-host transfers (e.g. `float(jax_array)`,
  `.item()`, `np.asarray()`): each one is a materialization point
  that breaks the JIT boundary and prevents the compiler from fusing
  across it. Ask whether materialization belongs in the protocol
  return type or in the caller after a batch of operations completes.
- What requires communication, AMR synchronization, host visibility,
  reduction-result visibility, or a profiling/timing boundary?
- Is the no-fusion/materialization boundary owned by the correct layer?

### 7. Backend / Lowering Trace

Describe what a backend author receives and what they are allowed to
change.

Check:

- What is the lowering boundary?
- What is guaranteed by the public contract?
- What is an implementation strategy?
- Which optimizations are allowed but not required?
- Does the design avoid per-element dynamic dispatch in compiled hot
  loops?

### 8. Alternative Failure Pass

Argue against the chosen design. The reviewer must answer these prompts
even if the design seems clean:

- Where does this abstraction collapse in a real domain workflow?
- Which concept is secretly doing another concept's job?
- What would be serialized unnecessarily?
- What ordering dependency would be missed?
- What optimization could change semantics if a boundary is implicit?
- What would a backend, driver, or physics author need that
  `ARCHITECTURE.md` does not specify?
- What simpler option was rejected, and is the rejection based on
  downstream cost rather than implementation effort?

### 9. Decision Delta

End with one of:

- **Ready:** the abstraction boundaries survived the stress review.
- **Ready with named risks:** the design is coherent, but specific
  risks must be tracked.
- **Needs revision:** the design has an ownership, dependency,
  ordering, or API-shape flaw that should be fixed before merge.

For "Ready with named risks" and "Needs revision", list the risks or
required changes concretely.
