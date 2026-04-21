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

This repo uses miniforge to provide a self-contained environment.
DO NOT use system Python or any external `python`/`pytest` command.

### Setup (One-Time)

If `environment/miniforge/` directory is missing:
```bash
bash scripts/setup_environment.sh
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

**If the env check fails** (script prints `✗ WRONG ENVIRONMENT` and
exits non-zero), stop immediately. The correct way to start a session
is:
```bash
./scripts/start_agent.sh claude   # or gemini / codex
```
`start_agent.sh` activates the environment automatically before
launching the agent. Alternatively, activate manually:
```bash
source scripts/activate_environment.sh
```

**If the env check passes but either follow-up check fails**, re-run
`setup_environment.sh` or the remediation commands printed by the
script — both are needed so `pre-commit run --all-files` works
locally before pushing (see *Branches and PRs*).

If the session starts from a parent organization workspace, run the
script from the `cosmic-foundry` checkout directory.

---

## Roadmap position

**At the start of every session**, read `## Current work` in `ARCHITECTURE.md`.
It is the immediate implementation queue. The rest of `ARCHITECTURE.md` covers
the long-horizon capability sequence (epochs, milestones, verification
standard) — items there are not yet specified well enough to implement
without further design discussion. When an item becomes fully specified
and unblocked, move it into `## Current work`.

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
items in `## Current work`** and ask two questions for each:
1. Does it have enough detail to be implementable without further design discussion?
2. Is anything in the current change inconsistent with it?

If the answer to (1) is no, flesh out the missing details in `ARCHITECTURE.md`
in the same PR (moving the item from later in the file if it lives there).
Do not update `ARCHITECTURE.md` speculatively — it records live decisions
only and is updated by the PR that implements the change. If the answer
to (2) is yes, resolve the inconsistency before merging. The goal is that
the next item is always fully specified before the current one lands.

---

## Epoch retrospective

**When an epoch is declared complete**, before opening any code PR for
the next epoch, perform a retrospective review. The retrospective
produces only documentation changes — ARCHITECTURE.md updates, roadmap updates,
process document corrections. It does not introduce code changes or
new features. Any code issue discovered during the retrospective
becomes a separate PR with its own spec and tests. The question the
retrospective asks is: *what did we learn during this epoch that
should update our plans?*

The retrospective covers:

1. **`ARCHITECTURE.md`**. Did the implementation reveal that any live
   decision needs updating? Were any open questions resolved? Are any
   paragraphs now self-evident from the code and therefore removable?

2. **Epoch sequence and current work.** Does the upcoming simulation or
   V&V scope still make sense given what we built? Should any epoch
   one-liner be reworded, reordered, or split? Is the current work
   section still accurate?

3. **Process documents** (`DEVELOPMENT.md`, `ARCHITECTURE.md`). Did any
   development rules prove unworkable, insufficient, or in need of
   precision?

4. **Surprises and pain points**. What was harder than expected? What
   design decisions caused rework? What would have been better to
   decide earlier? Capture these as updates to `ARCHITECTURE.md` or
   `DEVELOPMENT.md` — wherever the lesson is most actionable for the
   next epoch.

The output is one or more documentation PRs landing before Epoch N+1
code begins.

---

## Architectural Decisions

Architectural decisions live in `ARCHITECTURE.md`. Each live decision
is a one-paragraph claim. When a decision is made, add a paragraph in
the appropriate section. When a decision is superseded by the code or
withdrawn, remove it. When an open question is resolved, move it from
the *Open questions* section to the appropriate live section and update
the affected modules.

Any architecture review should also include a subjective pass over the
`*(Current inconsistency: ...)*` annotations in `ARCHITECTURE.md
§Architectural basis`. For each one, ask: has recent work resolved this
inconsistency? If yes, remove the annotation and close or update the
corresponding ARCHITECTURE.md gap-closure item.

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

Lanes B and C require a `_derive()` function in the production module with
executable SymPy checks on load-bearing algebraic steps. Infrastructure
capabilities (mesh topology, I/O, field placement) are out of scope for
lane classification.

The lane must be stated in the PR description, e.g. `Lane C (origination).
Reference papers: [...]`. For Lane B, explicitly record that the reference
source was not consulted.

---

## For AI agents

The following guidelines supplement the workflow rules above and apply
to all AI agents working on this repository, regardless of platform
(Claude Code, Codex, Gemini, or others).

### Platform role

Cosmic Foundry is the **organizational platform** for the simulation
ecosystem. Application repositories — covering stellar physics,
cosmology, galactic dynamics, planetary formation, and other domains
— build on top of it. The platform/application split is documented in
`ARCHITECTURE.md §Platform and application split`.

In practice this means:

- Reusable computation infrastructure (kernels, mesh, fields, I/O,
  diagnostics) belongs here.
- Domain-specific physics implementations and observational validation
  data belong in the relevant application repo.
- If a task spans the platform and an application repo, use separate
  branches and pull requests for each repository. Keep the platform
  change minimal and self-contained; the application repo change
  depends on it.
- Cross-scale workflows that compose two or more application domains
  belong in their own repository, not here.
- **This file is the authoritative source for all contributor and
  AI-agent behavior across the organization.** Application repo
  `DEVELOPMENT.md` files are intentionally thin: they delegate here
  for workflow rules and agent behavior, adding only what genuinely
  differs for that repo.

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

2. **Read `## Current work` in `ARCHITECTURE.md`** — current planned work
   and navigation anchor.

3. **Read `ARCHITECTURE.md`** — all live architectural decisions. When
   work touches a topic documented there, read it before making changes.

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
