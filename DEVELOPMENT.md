# Cosmic Foundry — Development Guide

This document covers the development workflow for all contributors
to this repository. For cross-cutting architectural decisions and open
design questions, see [`ARCHITECTURE.md`](ARCHITECTURE.md). For the
planned work and roadmap, see [`ARCHITECTURE.md`](ARCHITECTURE.md).

---

## Development Rules

This section is the authoritative quick-reference for branch and PR
discipline.

### Branches and PRs

- Every change lands via a pull request. **Never commit directly to
  `main`.** `main` is a read-only integration target; all work happens
  on topic branches.
- Create topic branches from `main`:
  ```bash
  git checkout -b feat/my-change main
  ```
- **After a PR merges**, pull main before starting the next branch:
  ```bash
  git checkout main && git pull origin main
  ```
- **Check PR state before pushing follow-up commits.** Verify a PR is
  still open before pushing to its branch:
  ```bash
  gh pr view --repo cosmic-foundry/cosmic-foundry \
    --json state --jq .state
  ```
  If the result is `MERGED` or `CLOSED`, do not push to that branch.
  Delete the local branch (`git branch -D <branch>`), pull main, and
  create a new topic branch for the new work.
- **Open pull requests** with `gh pr create`. Do not rely on
  `gh`'s default-repo inference — state it explicitly:
  ```
  gh pr create \
    --repo cosmic-foundry/cosmic-foundry \
    --base main
  ```
- CI's `pre-commit` job is a required status check; PRs cannot merge red.
- **Run `pre-commit run --all-files` before pushing.** CI runs the
  same hooks; catching failures locally avoids a round-trip. If the
  command is not found, the working copy's env is stale or the git
  hook was never installed — see *Environment → Before Any Work*.
### Pre-PR checklist

Before opening or pushing to a PR:

- [ ] Read `DEVELOPMENT.md` (this file) and `ARCHITECTURE.md` to understand the rules and decisions that govern the PR
- [ ] Align the built-in task plan with `ARCHITECTURE.md ## Current work`
- [ ] Run `pre-commit run --all-files` locally and fix any failures
- [ ] Read `## Current work` in `ARCHITECTURE.md` to understand the current planned work
- [ ] Determine if this PR completes any of the planned items
  - [ ] **If yes:** Remove the item from `ARCHITECTURE.md`
  - [ ] **If yes:** Horizon-scan the next items — are they fully specified? Flesh out details if needed
  - [ ] **If yes:** Verify no inconsistencies between this change and the next planned items
  - [ ] **If no:** Note "No change to roadmap position" in the PR description

### Commit size

- Code commits target approximately 150 lines of diff, with a
  soft ceiling of ~400 lines. Past the ceiling, split the commit
  or justify the size in the PR description.
- One logical change per commit — the LOC numbers are a proxy
  for reviewer cognitive load, not the rule itself.
- Generated files, lock files, fixtures / golden data, and pure
  deletions don't count toward the target or ceiling.
- Documentation diffs (research notes, roadmap edits,
  README / DEVELOPMENT.md / similar) are exempt from the guideline.

### History

- Never force-push a branch with an open PR or merged commits.
- Never alter merged history (no rebase, no `reset --hard`, no
  amending merged commits).
- One-off `git push --force-with-lease` on a pre-PR topic branch
  is allowed only with explicit user approval, for fixes that
  cannot be resolved with a forward commit (e.g. correcting author
  identity on a just-pushed commit).

### Durable metadata

- Never include local absolute filesystem paths (e.g. `/Users/…`,
  `/home/…`, `C:\Users\…`) in commit messages, PR titles, PR
  descriptions, or ARCHITECTURE.md. Use repository-relative paths or
  generic tool commands instead.
- Never commit API keys, tokens, or credentials. If one leaks,
  rotate it — rewriting history does not un-leak a pushed secret.

### Attribution

- AI-agent commits carry a `Co-Authored-By` trailer naming the
  agent and model.
- PR descriptions disclose AI-agent involvement when an agent
  generated substantial content.

### Project status

- This project has not started versioning or published stable APIs
  yet. Do not preserve backwards compatibility by default during
  structural refactors.
- Do not encode the repository's current roadmap epoch in code,
  tests, configuration, or live documentation. Epoch labels belong
  in roadmap and planning text, not in module docstrings,
  pytest marker descriptions, API notes, or overview copy as a
  statement of current project state.

---

## Environment

Run `./scripts/agent_health_check.sh` at the start of every session. If it
fails, run `bash scripts/setup_environment.sh` to repair the environment.

## Test Harness

Tests are organized as claim registries.  A claim is a small object with a
description and one `check(...)` method; top-level pytest functions only
parametrize claim lists and call `claim.check(...)`.

The three standard axes are:

- `test_correctness`: semantic, algebraic, structural, or residual claims.
- `test_convergence`: mesh/refinement claims against analytical expectations.
- `test_performance`: roofline or cost-to-accuracy claims.

Scalable Tensor-backed claims receive `ExecutionPlan` from `tests/claims.py`.
The plan carries the chosen backend, device kind, device calibration, and
`CF_CLAIM_WALLTIME_BUDGET_S`.  Claims use `batch_size_for`,
`problem_size_for`, or `refinement_count_for` to choose the largest
conservative extent that fits the active budget.  The budget changes extent
only; it must not change the mathematical assertion.

Scalar/debug shapes remain appropriate for symbolic structure checks and for
control-flow-heavy assertions where batching obscures diagnostics.  Batched
claims must report failed-lane metadata with `BatchedFailure`: batch index,
method/order/problem identifiers, parameters, actual and expected values,
error, and tolerance.  Rerun one lane by setting `CF_TEST_BATCH_INDEX=<index>`;
this forces CPU execution for replay.

`tests/conftest.py` calibrates CPU and optional GPU Tensor rooflines at session
start.  GPU execution is selected only when a GPU backend calibrates
successfully and its compute-bound roofline clears the shared CPU/GPU trust
threshold.  Otherwise GPU-specific claims skip with the recorded reason and
all `ExecutionPlan` claims run on CPU.

`tests/test_tensor.py` is the only test file allowed to use raw NumPy or JAX
directly for parity, calibration, and roofline trust checks.  Other module
tests use `Tensor` for states, RHS values, residuals, norms, and batched
comparisons, materializing Python values only at the final assertion boundary.

---

## Roadmap position

**At the start of every session**, read `## Current work` in `ARCHITECTURE.md`.
It is the durable implementation queue. During agent-led work, keep it aligned
with the built-in task plan: the task plan is local execution state, and
`Current work` is the short-horizon record that survives the session.

The rest of `ARCHITECTURE.md` covers long-horizon capability direction. Treat
those sections as design context, not an implementation queue. Move an item
into `## Current work` only when it has become concrete enough for the next PR
or two.

Every PR should state whether it advances the simulation track, the
V&V track, or both. Cross-track dependencies must be explicit in the
PR description.

Maintenance and tooling PRs that do not advance either track should
note "No change to roadmap position" in the PR description.

**When a PR completes a planned item**, that same PR must remove the
corresponding entry from `ARCHITECTURE.md ## Current work`. Do not leave
completed items there as historical record — the git log serves that
purpose. The rule is: if the item is done when the PR merges, it is
gone from `ARCHITECTURE.md` when the PR merges.

**Before merging a completing PR, horizon-scan the next well-defined
items in `## Current work`** and ask three questions for each:
1. Does it have enough detail to be implementable without further design discussion?
2. Is anything in the current change inconsistent with it?
3. Does it still encode premises, or has it become an object-level conclusion?

If the answer to (1) is no, flesh out the missing details in `ARCHITECTURE.md`
in the same PR (moving the item from later in the file if it lives there).
If the answer to (2) is yes, resolve the inconsistency before merging. If the
answer to (3) is no, rewrite or delete the item. Do not update
`ARCHITECTURE.md` speculatively: it records live decisions and the next one to
three tasks, not a backlog.

---

## Development loop

Develop mathematical infrastructure by starting from a concrete calculation,
then working backward to the premise that makes the calculation meaningful.
The loop is not "add capability, add test, add docs." It is "name the
calculation, identify the premise, encode it once, make the consequences
discovered, then delete anything that was only recording the consequence by
hand."

Every PR must walk the full loop. A PR proposal starts by naming the concrete
calculation claim it will deliver. If the calculation cannot be named, the PR
is not yet well-posed. A structural change, schema change, documentation
change, or deletion is not complete until it grounds out in that calculation
claim in the same PR. Prefer extending an existing real-ish problem when that
is the shortest honest path. If the change cannot be connected to a
calculation without inventing fake work, stop and discuss the premise instead
of opening a partial PR.

The required loop is:

1. **Name the calculation.** Choose the concrete workload that should pass
   after the PR: a synthetic reaction network, transient
   diffusion/advection-diffusion problem, implicit step, conservation
   projection, or similarly meaningful case.
2. **Generalize the premise.** Ask what mathematical fact would make that
   calculation non-accidental, then recursively generalize until another
   generalization would make the requirement less precise, less testable, or
   harder for a human to read.
3. **Encode the premise.** Improve schemas, ownership models, descriptor
   construction, theory objects, or capability projections by representing
   the fact from which the desired behavior follows.  When two independent
   grounded calculations share the same premise, ask whether the premise is a
   type relationship: if objects share structural facts, descriptor
   projections, or admissible algorithms, prefer making that relationship
   explicit in the class or protocol hierarchy.  Helpers may construct
   instances, but they must not hide the ontology that the calculations have
   earned.
4. **Make discovery do the work.** New implementations, claims, and atlas
   regions should become subject to the relevant requirements by inheritance,
   structure, schema registration, or implemented methods; editing a list by
   hand is a temporary smell unless the list is itself the mathematical object.
5. **Run the calculation.** Add or improve the calculation claim named in step
   1. It should leave behind at least one durable formal result: a descriptor
   coordinate, structural invariant, generated atlas region, numerical claim,
   discovered structural test, or deletion of a misleading abstraction.
6. **Correct the model by deleting.** Delete names, fields, categories,
   coverage claims, object-level test cases, docs, or examples that the
   calculation shows are fake, redundant, stale, or insufficiently precise.

After each real-ish problem, map the new premise back to `theory/` and ask
whether the parameter schema or computation code has started to encode a
theory object implicitly. One grounded occurrence stays test-local. Two
independent occurrences trigger a theory review. Three occurrences mean the
codebase is probably maintaining a computational shadow of theory: either
promote the minimal object into `theory/` or delete the duplicated
representation. The promotion target is the premise, not the application
projection; for example, a conserved finite-state transformation may be the
theory object, while a reaction network remains only one computational view of
it.

Apply the same discipline to tests. Test lines are cheaper than source lines,
not free. Periodically review claim registries for repeated examples,
object-level regressions, and fixtures that now encode the same premise. One
grounded test can remain as evidence. Two similar tests should prompt a search
for the discovered invariant that covers both. Three similar tests require
consolidation: replace them with one parameterized or auto-discovered claim, or
delete the weaker cases. A growing test count is acceptable only when it grows
the set of formal premises being checked, not when it grows an inventory of
examples.

### Design principles

When changing computation capabilities, apply these constraints in order:

1. **Encode premises, not conclusions.** Do not encode labels such as
   `explicit`, `SPD`, `least_squares`, or `nonlinear` as independent truths
   when they can be derived from equations, inheritance, operators, residuals,
   constraints, or available oracles.
2. **Prefer symbolic or numeric requirements over string literals.** Strings
   are acceptable as rendering labels, external protocol values, or
   human-facing text. They should not identify mathematical requirements when
   an enum, type, predicate, dimension, tolerance, or structural relation can
   do it.
3. **Formalize requirements as executable claims.** A requirement belongs in
   code only when a test can check it mechanically. Prefer one discovered claim
   over several named examples.
4. **Make tests automatically discovered.** Tests should discover
   implementations, schemas, claims, and ownership regions from the code they
   govern. Adding a new implementation should usually make it subject to
   relevant structural tests without editing a list by hand.
5. **Use AST tests for meta-level antipatterns.** Structural tests should ban
   the general shape of the invalid program, not a named symbol, field, file,
   or embarrassing historical instance. If the test needs string matching to
   find the bad thing, generalize the premise again.
6. **Delete before adding.** If two abstractions overlap, first ask what
   precise scenario distinguishes them. If the distinction cannot be expressed
   mathematically or operationally, delete or merge one of them.
7. **Keep the human projection subordinate.** Human-facing categories, atlas
   titles, filenames, captions, and package names are projections of the
   mathematical model. They must not become parallel sources of truth.
8. **Make earned hierarchy visible.** Descriptor schemas should project type
   relationships, not compensate for their absence. Do not invent a grand base
   class speculatively, but once repeated calculations expose a shared premise,
   failing to represent it in the type system is a code smell.
9. **Keep the code readable.** Mathematical precision is not permission to hide
   intent. Prefer small enums, predicates, dataclasses, and helper functions
   whose names expose the underlying relation.

### Planning discipline

The built-in task plan and `ARCHITECTURE.md ## Current work` must stay aligned
during active development.

- Keep `Current work` to the next one to three concrete tasks.
- Update the local task plan when the active step changes; update
  `Current work` in the same PR when the short-term horizon changes.
- When a PR completes a current-work item, remove that item in the same PR.
- Add only the next visible task or two; do not over-plan beyond what the latest
  implementation has made clear.
- If a real-ish problem exposes a schema defect, update `Current work` toward
  the correction rather than continuing the previous abstraction path blindly.

---

## Epoch retrospective

When an epoch is declared complete, open a documentation-only PR before any
code PR for the next epoch. Update `ARCHITECTURE.md` (close the epoch row in
the table, update `## Current work`, record any open questions resolved) and
`DEVELOPMENT.md` (any process rules that proved unworkable). No code changes;
any code issue discovered becomes a separate PR.

---

## Physics capability implementation paths

Every PR that adds or changes a physics capability is classified into one
of three lanes. The classification matters for two reasons: licensing (many
astrophysics reference codes carry copyleft terms that would propagate to
this codebase if their source were consulted) and rigor (Lanes B and C
require machine-checkable derivations that Lane A defers to the reference).

- **Lane A — Port-and-verify.** A permissively-licensed reference
  implementation exists (MIT, BSD, Apache, or similar). Adapt it with
  attribution. No derivation document required; the reference source is
  openly inspectable and serves as the derivation.

- **Lane B — Clean-room from paper.** The only reference implementations
  are copyleft-licensed (GPL and similar — common among major astrophysics
  codes). The reference source tree **must not be opened**: no `git clone`,
  no source browsing, no cached previews. Work from papers and vendor
  documentation only. A derivation document is required to demonstrate
  independence from the copyleft source.

- **Lane C — First-principles origination.** No reference implementation
  to port, or the goal is to generalize, extend, or understand the formalism
  beyond what any specific reference provides. Derive from first principles.
  A derivation document is required; principled disagreements with the
  literature are recorded inside it.

Lanes B and C require machine-checkable derivations.  The default is
always to generalize verification into an auto-discoverable framework
rather than write a one-off test function.  Writing a `test_*` function
is a signal that generalization has not been done yet; when that happens,
the right response is to extend the framework so the check applies to the
whole class of objects, not just the instance in front of you.

**Convergence order verification.** Every concrete `DiscreteOperator`
subclass (e.g. `DiffusiveFlux`) that claims a convergence order must be
covered by the convergence harness. The structure:

- The class declares `order: int` (inherited from `DiscreteOperator`) and
  `min_order: ClassVar[int]` + `order_step: ClassVar[int]` at the class
  level — the validity range of the `order` parameter.
- The class carries `continuous_operator: DifferentialOperator` — the
  continuous operator it approximates.  The convergence test auto-computes
  the exact value as `Rₕ(L φ)` using `CartesianRestrictionOperator` and
  `continuous_operator`; no per-class oracle is needed.
- `tests/test_convergence_order.py` is the single parametric test: it
  verifies the error polynomial has zeros at `h⁰…h^{p-1}` and a nonzero
  `h^p` leading term, where `p = instance.order`.  One test function covers
  all convergent classes.  No additional `test_*` functions are written for
  convergence.
- Legacy inventories such as `_INSTANCES` are bookkeeping, not an architectural
  pattern to copy. When touching them, prefer moving the harness toward
  structural discovery; if a manual inventory remains, keep it local and make
  the tested premise explicit.

Infrastructure capabilities (mesh topology, I/O, field placement) are
out of scope for lane classification.

The lane must be stated in the PR description, e.g. `Lane C (origination).
Reference papers: [...]`. For Lane B, explicitly record that the reference
source was not consulted.

---

## For AI agents

The following guidelines supplement the workflow rules above and apply
to all AI agents working on this repository, regardless of platform
(Claude Code, Codex, Gemini, or others).

### Session startup

**At the start of every session**, in this order:

1. **Run the health check:**
   ```bash
   ./scripts/agent_health_check.sh
   ```
   The script verifies that (a) the `cosmic_foundry` conda environment
   is active, (b) `pre_commit` is importable, and (c) the git
   pre-commit hook is installed.

   **If the env check fails** (script prints `✗ WRONG ENVIRONMENT` and
   exits non-zero), stop immediately and warn the user:

   > ⚠️ The `cosmic_foundry` conda environment is not active. All
   > Python commands in this repo (`python`, `pytest`, `mypy`,
   > `pre-commit`, `sphinx-build`) must run inside this environment.
   > Using the wrong environment causes silent misconfiguration errors.
   >
   > The correct way to start an agent session is:
   > ```bash
   > ./scripts/start_agent.sh claude   # or gemini / codex
   > ```
   > `start_agent.sh` activates the environment automatically before
   > launching the agent. Do not proceed until the user confirms the
   > session has been restarted this way, or manually activates the
   > env:
   > ```bash
   > source scripts/activate_environment.sh
   > ```
   > then re-launches the agent from that shell.

   **If the env check passes but either follow-up check fails**, re-run
   `setup_environment.sh` or the remediation commands printed by the
   script.

2. **Read `## Startup context` in `ARCHITECTURE.md`** — navigation anchor
   for loading only the architecture sections relevant to the task.

3. **Read `## Current work` in `ARCHITECTURE.md`** — current planned work
   and implementation queue.

4. **Read targeted `ARCHITECTURE.md` sections only as needed.** When work
   touches a topic documented there, read the governing anchored section
   before making changes. Do not duplicate architectural facts in a second
   startup summary; `## Startup context` is an index, not a source of truth.

### Physics lane selection

The three lanes (A, B, C) are defined in
[§Physics capability implementation paths](#physics-capability-implementation-paths).

For any task that touches a physics capability:

1. **Classify the lane.** Determine whether a reference implementation
   exists and check its license. If the license is permissive, Lane A
   applies. If the only references are copyleft-licensed, Lane B is
   mandatory. If no reference implementation exists, or the user's
   framing implies generalization or novel work ("extend," "generalize,"
   "give us our own version of X"), Lane C applies.
2. **Propose the derivation-first lane when it appears to apply.**
   If the default would be Lane A but the task framing suggests Lane C,
   propose Lane C to the user before writing code. If the reference is
   copyleft, Lane B is not a proposal — state it as the required lane
   and confirm the user agrees before proceeding.
3. **Record the lane in the PR description** in the first paragraph,
   e.g. `Lane C (origination). Reference papers: [...]`. For Lane B,
   explicitly record that reference source was not consulted.
4. **When uncertain, propose the derivation-first lane (B or C,
   whichever fits) and ask the user to confirm** rather than defaulting
   silently to Lane A.

The lane choice is the user's decision; the agent's job is to surface
the decision transparently, not to make it silently.

### Weighing architectural options

You are an AI agent. Writing code and prose costs you nothing. This
means implementation effort is not a meaningful criterion when
comparing architectural options — it is a rounding error, not a
trade-off. The costs that actually matter are all downstream:

- reviewer cognitive load,
- ongoing operational and maintenance burden,
- reversibility if the choice turns out to be wrong,
- correctness and safety guarantees,
- blast radius of a failure.

Rank options by these. Include the simpler option in every comparison
even when you intend to recommend the richer one — the user needs the
full option space to make an informed decision.

### Code economy

Source code and documentation lines are expensive. Every line must be
read, understood, and kept accurate by every future contributor. Each
abstraction, helper, and prose paragraph is a permanent cognitive tax.
When in doubt, delete rather than retain; the simpler the codebase, the
lower the maintenance burden and the harder it is for bugs to hide.

Test lines are cheaper than source lines, but they are not free. A good test
encodes a premise once and discovers every object governed by that premise. A
weak test enumerates conclusions, names past mistakes, or hard-codes examples
that future contributors must maintain by hand.

In practice: before adding a helper, ask whether the call site is already
readable without it. Before adding a docstring sentence, ask whether a
good name already says it. Before adding a test case, ask whether the harness
can discover the whole class of cases instead.

---

## Docstring conventions

Docstrings serve multiple purposes: establishing the mathematical contract,
educating readers without formal training, and guiding practical use. Structure
docstrings in three parts, in order:

**Part 1: Formal mathematical definition.** State what the object IS and DOES
using mathematically rigorous language. Name function spaces, norms, and types
explicitly. This is the contract reviewers check.

**Part 2: Informal explanation.** Restate what Part 1 says at the level of a
bright physics graduate student (strong practical calculus, no formal
functional analysis assumed). Remove the heavy notation; explain the intuition.

**Part 3: Practical context.** How is this used in the code? Why does this
choice matter? What breaks if someone misuses it? When should I use this vs.
a simpler alternative?

Example:

```python
def inner_product_L2h(u, v, mesh):
    """Discrete L² inner product on mesh cell averages.

    ⟨u, v⟩_h := Σᵢ |Ωᵢ| uᵢ vᵢ, where |Ωᵢ| is the cell volume and
    uᵢ, vᵢ are cell-average values. This is the norm in which FVM convergence
    proofs are stated; discrete operators are symmetric and positive-definite
    with respect to this pairing, not the Euclidean dot product.

    In plain terms: multiply corresponding values at each cell, scale by the
    cell's volume, and sum. This gives more weight to larger cells, which
    matches the physics: a large cell has more influence on the solution.

    Use this when verifying operator symmetry or testing convergence. DO NOT
    use numpy's dot product — the cell volumes must be included. If the mesh
    is uniform (constant Δx), the two norms are proportional; non-uniform
    meshes expose the difference.
    """
```

**Present tense only.** Describe current design, not historical decisions.
"This replaces the raw kwargs-passing from version 2" belongs in the commit
message. Docstrings say "This takes a Domain object to enforce type safety."
