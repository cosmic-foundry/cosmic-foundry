# Agent Instructions

These guidelines apply to all AI agents working on this repository,
regardless of platform (Claude Code, Codex, Gemini, or others).

## Platform role

Cosmic Foundry is the **organizational platform** for the simulation
ecosystem. Application repositories — covering stellar physics,
cosmology, galactic dynamics, planetary formation, and other domains
— build on top of it. See
[ADR-0014](adr/object-level/ADR-0014-platform-application-architecture.md) for the
authoritative split.

In practice this means:

- Reusable computation infrastructure (kernels, mesh, fields, I/O,
  diagnostics) and manifest infrastructure (`cosmic_foundry.manifests`:
  HTTP client, `ValidationAdapter`, `Provenance`, base schemas) belong
  here.
- Domain-specific physics implementations and observational validation
  data belong in the relevant application repo.
- If a task spans the platform and an application repo, use separate
  branches and pull requests for each repository. Keep the platform
  change minimal and self-contained; the application repo change
  depends on it.
- Cross-scale workflows that compose two or more application domains
  belong in their own repository, not here.
- **This file is the authoritative source for all software design
  direction across the organization.** Application repo `AI.md` files
  are intentionally thin: they delegate here for development rules,
  commit discipline, ADR process, physics capability lanes, code quality
  standards, and architectural weighing criteria. They add only what
  genuinely differs for that repo (fork/PR targets, environment setup,
  domain-specific data rules). If a software design rule or agent
  guideline belongs to both the platform and an application repo,
  it belongs here — not in both places.

## Development Rules

The authoritative source is
[ADR-0005](adr/meta-level/ADR-0005-branch-pr-attribution-discipline.md). This
section is the informal quick-reference; when the two disagree, the
ADR wins.

### Branches and PRs

- Only work on a fork of the upstream repository.
- Every change lands via a pull request; no direct commits to
  `upstream/main`.
- Create topic branches from `origin/main` (the fork's main), not
  from `upstream/main` directly. Syncing `origin/main` to
  `upstream/main` is an explicit separate step.
- **Syncing `origin/main` after a merge.** There is no named
  `upstream` remote in this repo (omitted to prevent accidental
  pushes). Use `gh repo sync` instead:
  ```bash
  gh repo sync cosmic-foundry-development/cosmic-foundry \
    --source cosmic-foundry/cosmic-foundry --branch main
  git checkout main && git pull origin main
  ```
- **Check PR state before pushing follow-up commits.** The user
  sometimes merges a PR without telling the agent, and sometimes
  forgets to sync `origin/main` afterward. Before pushing additional
  commits to a branch that already has a PR, verify the PR is still
  open:
  ```bash
  gh pr view --repo cosmic-foundry/cosmic-foundry \
    --json state --jq .state
  ```
  If the result is `MERGED` or `CLOSED`, do not push to that branch.
  Instead: delete the local branch if it exists (`git branch -D
  <branch>`), sync `origin/main` (see above), check out `main`, and
  create a new topic branch for the new work. Also check whether
  `STATUS.md` lists this PR as `Open` and update the entry to
  `Merged` if so.
- **Open pull requests against `upstream/main`**, not against the
  fork. Push the topic branch to `origin`, then open the PR so it
  merges into the upstream repository's `main`. Do not rely on
  `gh`'s default-repo inference — state it explicitly:
  ```
  gh pr create \
    --repo cosmic-foundry/cosmic-foundry \
    --base main \
    --head <fork-owner>:<topic-branch>
  ```
- CI's `pre-commit` job is a required status check on
  `upstream/main`; PRs cannot merge red.
- **Run `pre-commit run --all-files` before pushing.** CI runs the
  same hooks; catching failures locally avoids a round-trip. If the
  command is not found, the working copy's env is stale or the git
  hook was never installed — see *Environment → Before Any Work*.
- **PR review.** The project's adversarial reviewer lives at
  [`pr-review/`](pr-review/README.md) (roles + checklist of
  historical failure modes). Run the reviewer against any non-trivial
  PR before requesting human review. For ADRs and architecture-changing
  PRs, also run the architecture stress-review checklist at
  [`pr-review/architecture-checklist.md`](pr-review/architecture-checklist.md).

  **For ADR PRs the architecture stress-review note is required, not
  optional.** Include it in the PR description before opening. A PR
  that adds or amends an ADR without a stress-review note should not
  be opened for human review — the note is evidence that the review
  was actually done, not a post-hoc summary.

  **Terminal / automation:**
  ```bash
  ./scripts/review_pr_with_claude.sh <n>   # Claude Code
  ./scripts/review_pr_with_codex.sh <n>    # Codex
  ./scripts/review_pr_with_gemini.sh <n>   # Gemini CLI
  ```
  Set `COSMIC_FOUNDRY_PR_REPO` to override the default repository
  (`cosmic-foundry/cosmic-foundry`).

  **In-session requests:**
  Inside an active session, treat user requests of the form "Review PR N"
  or "Review N" as a request to run the adversarial reviewer:
  1. Read `pr-review/agent.md` and `pr-review/checklist.md` in full.
     If the PR is architecture-changing, also read
     `pr-review/architecture-checklist.md`.
  2. Fetch PR metadata and diff:
     `gh pr view N --repo cosmic-foundry/cosmic-foundry`
     `gh pr diff N --repo cosmic-foundry/cosmic-foundry`
  3. Perform the review using the fetched data and the working tree
     (read-only inspection only).
  4. Return the report in the exact format required by `pr-review/agent.md`.

### Commit size

- Code commits target approximately 150 lines of diff, with a
  soft ceiling of ~400 lines. Past the ceiling, split the commit
  or justify the size in the PR description.
- One logical change per commit — the LOC numbers are a proxy
  for reviewer cognitive load, not the rule itself.
- Generated files, lock files, fixtures / golden data, and pure
  deletions don't count toward the target or ceiling.
- Documentation diffs (ADRs, research notes, roadmap edits,
  README / AI.md / similar) are exempt from the guideline.

### Weighing architectural options

When comparing architectural options (in ADRs, design discussions,
or ad-hoc recommendations), do **not** weight by author effort or
lines of code produced. Agent-assisted authoring has made upstream
writing cost a rounding error; downstream costs now dominate and are
what the analysis should rank by:

- reviewer cognitive load (the existing ~150-line commit guideline
  is also a proxy for this),
- ongoing operational and maintenance cost,
- reversibility if the choice turns out wrong,
- correctness and safety guarantees,
- blast radius of a failure.

Implementation effort is a tiebreaker at most. Present options to
the user in terms of these downstream costs, not in terms of "cheap
vs. expensive to build." Always include the lower-complexity option
in the comparison even when recommending a richer one — the goal is
to keep the decision with the user, not to bias toward ambition just
because ambition is cheap to type.

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
  descriptions, or ADR text. Use repository-relative paths or
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
  in roadmap and ADR planning text, not in module docstrings,
  pytest marker descriptions, API notes, or overview copy as a
  statement of current project state.

## Environment

This repo uses miniforge to provide a self-contained environment.
DO NOT use system Python or any external `python`/`pytest` command.

### Setup (One-Time)

If `miniforge/` directory is missing:
```bash
bash environment/setup_environment.sh
```

### Before Any Work

**Run the startup health check** immediately at the start of every
session:
```bash
./scripts/agent_health_check.sh
```

The script verifies three things in order:
1. `cosmic_foundry` conda environment is active
2. `pre_commit` Python package is importable
3. `.git/hooks/pre-commit` is installed

It is the only bash command auto-approved by the repository's
`.claude/settings.json`; the trust surface is exactly this committed
script and nothing else.

The auto-approval is Claude Code-specific because only Claude Code
supports exact-match Bash allowlists. Codex has no per-command
allowlist primitive, and Gemini's `tools.allowed` is prefix-match
(it would also auto-approve any command starting with the script
path, including chained or arg-appended invocations). Rather than
weaken the trust surface for those agents, the shortcut is omitted
— Codex and Gemini operators approve the health check once per
session the normal way.

**If the env check fails** (script prints `✗ WRONG ENVIRONMENT` and
exits non-zero), stop immediately and warn the user:

> ⚠️ The `cosmic_foundry` conda environment is not active. All Python
> commands in this repo (`python`, `pytest`, `mypy`, `pre-commit`,
> `sphinx-build`) must run inside this environment. Using the wrong
> environment causes silent misconfiguration errors.
>
> The correct way to start an agent session is:
> ```bash
> ./scripts/start_agent.sh claude   # or gemini / codex
> ```
> `start_agent.sh` activates the environment automatically before
> launching the agent. Do not proceed until the user confirms the
> session has been restarted this way, or manually activates the env:
> ```bash
> source environment/activate_environment.sh
> ```
> then re-launches the agent from that shell.

**If the env check passes but either follow-up check fails**, re-run
`setup_environment.sh` or the remediation commands printed by the
script — both are needed so `pre-commit run --all-files` works
locally before pushing (see *Branches and PRs*).

If the session starts from a parent organization workspace, run the
script from the `cosmic-foundry` checkout directory.

## Roadmap position

**At the start of every session**, read `STATUS.md` in the repository
root. It records the current object-level and meta-level positions,
recently completed milestones, and what work is next. This is the
fastest way to orient without reading the full roadmap planes.

Roadmap documentation is split onto two planes:

- `roadmap/object-level/README.md` — what platform and simulation capabilities
  the codebase is building.
- `roadmap/meta-level/README.md` — how the project verifies, validates,
  regenerates, and audits object-level claims.

Every PR should state whether it advances the object-level track, the
meta-level track, or both. Cross-track dependencies must be explicit in
the PR description and in any `STATUS.md` update.

When a PR completes a milestone or starts a new epoch, update `STATUS.md`
as part of that PR — including marking the PR's own row `Merged` and
updating *Next planned work*. Because the `STATUS.md` edit is committed
inside the PR, it lands atomically on merge, so the file is correct the
moment the PR closes. The only valid status values in the progress table
are `Planned` and `Merged`; `Open` is never correct to write, since a PR
marks itself `Merged` before it can be merged. Maintenance and tooling PRs that do not advance the
roadmap should note "No change to roadmap position" in both the PR
description and `STATUS.md` (or omit the `STATUS.md` edit entirely if
nothing changed).

After each PR merges, re-examine the relevant track's implementation
plan (see *Implementation plans* below). If the merged work changes the
sequence — new ordering constraints discovered, a planned PR split or
merged, scope added or dropped — update the plan in the same PR that
closes the milestone. If the resequencing is cross-cutting and
independent of the code change, open a standalone `docs(roadmap):` PR
instead. Simple tick-offs (marking an item complete) can travel with the
`STATUS.md` update.

## Implementation plans

**At the start of a new object-level epoch** (after retrospective PRs
land and before the first code PR opens), open one documentation PR that
appends an **Implementation plan** section to the epoch's roadmap file
(`roadmap/object-level/epoch-NN-*.md`). For meta-level stages, add or
update a dedicated implementation plan such as
`roadmap/meta-level/reproducibility-meta-generator.md`. The plan is a
numbered list of proposed PRs with:

- One-line scope per entry
- Explicit "depends on" notes for ordering constraints
- No detailed spec — enough to unblock the next 3–5 sessions

The plan is a living document. After each PR merges, re-examine
the next 3–5 entries and update the plan if the merged work changed
the picture (see *Roadmap position* above for when to do this inline
vs. as a standalone PR).

Mark completed items with ✓ and the PR number. Do not delete them
— the completed list is the running record of what was actually
built, and it feeds the next epoch retrospective.

## Epoch retrospective

**When an epoch is declared complete**, before opening any code PR for
the next epoch, perform a retrospective review. This is a read-only
survey of the whole repository — no code changes, no new features — that
asks: *what did we learn during this epoch that should update our plans?*

The retrospective covers:

1. **ADRs in force** (`adr/README.md` and each linked file). Did the
   implementation reveal that any decision needs updating? Were any
   "anticipated extensions" or "deferred" items resolved in practice?
   Were any stated consequences wrong?

2. **ADR set as a whole.** Beyond reviewing each ADR individually,
   examine the set itself. Are any two ADRs covering the same concern
   and worth combining? Has an ADR's scope drifted so it now overlaps
   with a newer one, and should be narrowed or merged into it? Are
   there numbering or ordering choices that would read more coherently
   if restructured? The goal is to keep the ADR family close to an
   *orthogonal basis* — each ADR covering one independent
   architectural concern, and the set as a whole the minimum number
   of decisions needed to explain the architecture. Propose
   reorganizations (combine, split, renumber, retire) as follow-up
   PRs rather than inline retrospective edits; the retrospective
   surfaces the need, the PR executes the change.

3. **Roadmap files** (`roadmap/object-level/README.md`,
   `roadmap/meta-level/README.md`, and the relevant per-epoch or
   meta-stage implementation plans). Does the upcoming object-level or
   meta-level scope still make sense given what we built? Are the design
   prerequisites still the right ones? Are the exit criteria still
   well-defined?

4. **`replication/` specs and formulas register**. Do the capability
   specs reflect what was actually implemented? Are there formula entries
   or capability stubs that should be updated now rather than left stale?

5. **`AI.md` and process documents**. Did any development rules prove
   unworkable, insufficient, or in need of precision?

6. **Surprises and pain points**. What was harder than expected? What
   design decisions caused rework? What would have been better to decide
   earlier? Capture these as ADR edits, roadmap notes, or additions
   to `AI.md` — wherever the lesson is most actionable for the next epoch.

The output is one or more PRs amending affected documents before Epoch N+1
code begins. These are documentation-only PRs; the retrospective itself
does not introduce code. If a surprise reveals a design mistake that
requires changing existing code, that is a separate PR with its own
spec and tests, not part of the retrospective sweep.

## Architectural Decisions

Architectural decisions are recorded as ADRs in `adr/`. **At the start
of every session**, read `adr/README.md` — it is the canonical registry
of every ADR in force and routes to `adr/object-level/README.md` and
`adr/meta-level/README.md`. When work touches a topic listed there, read
the relevant architecture plane and the full ADR before making changes;
the registry and plane documents are pointers, not summary substitutes.

When making a new architectural decision, copy
`adr/adr-template.md` to
`adr/<object-level|meta-level>/ADR-NNNN-<short-title>.md` and add a line
to `adr/README.md` in the same PR.

Before treating an architectural decision as ready for human review,
run `pr-review/architecture-checklist.md`. The checklist forces an
agent or reviewer to map the design space, define concept ownership,
write realistic usage traces, normalize dependencies and lowering
boundaries, and identify fences / materialization points. Include the
stress-review result in the ADR, the PR description, or the review
report.

ADRs describe current architecture. When a conversation implies an
ADR should change, propose the edit directly. If a decision is
entirely withdrawn, remove it from the index. See
[ADR-0005 §Decision → ADR editing policy](adr/meta-level/ADR-0005-branch-pr-attribution-discipline.md#adr-editing-policy).

## Physics capability implementation paths

Per [ADR-0013](adr/meta-level/ADR-0013-derivation-first-lane.md), every PR that
adds or changes a *physics capability* (as defined in ADR-0007
§Decision) is in one of three lanes:

- **Lane A — Port-and-verify.** Adapt from a permissively-licensed
  reference code with attribution. Default for permissive references.
  No derivation document required.
- **Lane B — Clean-room from paper.** Mandatory when the only
  reference is copyleft-licensed (per
  [`research/06-12-licensing.md`](research/06-12-licensing.md):
  GADGET-4, RAMSES, MESA, SWIFT, PLUTO, Arepo, ChaNGa, GIZMO-public,
  MPI-AMRVAC, BHAC, koral_lite, Dedalus, ...). The reference source
  tree **must not be opened** — no `git clone`, no source browsing,
  no cached previews with source content. Papers and vendor
  documentation only. A derivation document is required.
- **Lane C — First-principles origination.** For generalizations,
  extensions, and novel work where the goal is to *understand* the
  formalism rather than reproduce a specific reference. A derivation
  document is required; principled disagreements with the literature
  are recorded inside it.

Lanes B and C require a derivation document under `derivations/`
with executable SymPy checks on load-bearing algebraic steps.
Infrastructure capabilities (dispatch, mesh topology, I/O, field
placement) are out of scope.

### Agent responsibility per task

For any task that touches a physics capability:

1. **Classify the lane.** Look up the reference code's license in
   `research/06-12-licensing.md`. If no reference code exists, or
   the user's framing implies generalization or novel work
   (phrases like "extend," "generalize," "we might need to break
   new ground," "give us our own version of X"), Lane C applies.
2. **Propose the derivation-first lane when it appears to apply.**
   If the default would be Lane A but the task framing suggests
   Lane C, propose Lane C to the user before writing code. If the
   reference is copyleft, Lane B is not a proposal — state it as
   the required lane and confirm the user agrees before
   proceeding.
3. **Record the lane in the PR description** in the first
   paragraph, e.g. `Lane C (origination). Reference papers: [...]`.
   For Lane B, explicitly record that reference source was not
   consulted.
4. **When uncertain, propose the derivation-first lane (B or C,
   whichever fits) and ask the user to confirm** rather than
   defaulting silently to Lane A.

The lane choice is the user's decision; the agent's job is to
surface the decision transparently, not to make it silently.

## Code Quality

- Write code that is:
  - Self-documenting (clear variable names, simple logic)
  - Minimal (only what's needed, no over-engineering)
  - Testable (existing tests must pass, new features need tests)
