# Cosmic Foundry — Development Guide

This document covers the development workflow for all contributors
to this repository, human or agent. For AI-agent-specific session
instructions, see [`AI.md`](AI.md). For cross-cutting architectural
decisions and open design questions, see [`ARCHITECTURE.md`](ARCHITECTURE.md).
For the current status and planned work, see [`STATUS.md`](STATUS.md).

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
- **PR review.** The project's adversarial reviewer lives at
  [`pr-review/`](pr-review/README.md) (roles + checklist of
  historical failure modes). Run the reviewer against any non-trivial
  PR before requesting human review. For architecture-changing PRs,
  also run the architecture stress-review checklist at
  [`pr-review/architecture-checklist.md`](pr-review/architecture-checklist.md).

  **For architecture-changing PRs the stress-review note is required,
  not optional.** Include it in the PR description before opening. A
  PR that introduces or reshapes an architectural abstraction without a
  stress-review note should not be opened for human review — the note
  is evidence that the review was actually done, not a post-hoc
  summary.

  **Terminal / automation:**
  ```bash
  ./scripts/review_pr_with_claude.sh <n>   # Claude Code
  ./scripts/review_pr_with_codex.sh <n>    # Codex
  ./scripts/review_pr_with_gemini.sh <n>   # Gemini CLI
  ```
  Set `COSMIC_FOUNDRY_PR_REPO` to override the default repository
  (`cosmic-foundry/cosmic-foundry`).

### Pre-PR checklist

Before opening or pushing to a PR:

- [ ] Read `DEVELOPMENT.md` (this file) and `ARCHITECTURE.md` to understand the rules and decisions that govern the PR
- [ ] Run `pre-commit run --all-files` locally and fix any failures
- [ ] Read `STATUS.md` to understand the current planned work
- [ ] Determine if this PR completes any of the planned items
  - [ ] **If yes:** Remove the item from `STATUS.md`
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
  README / AI.md / similar) are exempt from the guideline.

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

**At the start of every session**, read `STATUS.md` in the repository
root. It is the immediate implementation queue. Read `ROADMAP.md`
for the long-horizon capability sequence (epochs, milestones, verification
standard) — items there are not yet specified well enough to implement
without further design discussion. When an item becomes fully specified
and unblocked, move it from `ROADMAP.md` to `STATUS.md`.

Every PR should state whether it advances the simulation track, the
V&V track, or both. Cross-track dependencies must be explicit in the
PR description.

Maintenance and tooling PRs that do not advance either track should
note "No change to roadmap position" in the PR description.

**When a PR completes a planned item in `STATUS.md`**, that same PR
must remove the corresponding entry from `STATUS.md` (or from the
`## Current work` list if it is a sequenced step). Do not leave
completed items in `STATUS.md` as historical record — the git log
serves that purpose. The rule is: if the item is done when the PR
merges, it is gone from `STATUS.md` when the PR merges.

**Before merging a completing PR, horizon-scan the next well-defined
items in `STATUS.md`** and ask two questions for each:
1. Does it have enough detail to be implementable without further design discussion?
2. Is anything in the current change inconsistent with it?

If the answer to (1) is no, flesh out the missing details in `STATUS.md`
in the same PR (moving the item from `ROADMAP.md` if it lives there). Do not update `ARCHITECTURE.md` speculatively
— it records live decisions only and is updated by the PR that implements
the change. If the answer to (2) is yes, resolve the inconsistency before
merging. The goal is that the next item is always fully specified before
the current one lands.

---

## Implementation plans

The immediate next work for both tracks is maintained in the
`## Current work` section of `STATUS.md`. Keep it current:
when an item completes, remove it from `STATUS.md`; when new immediate
work becomes fully specified and unblocked, move it from `ROADMAP.md`
to `STATUS.md`. Do not add items to `STATUS.md` that depend on decisions
not yet made — those belong in `ROADMAP.md`.

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

2. **`ROADMAP.md`**. Does the upcoming simulation or V&V scope still
   make sense given what we built? Should any epoch one-liner be
   reworded, reordered, or split? Is the immediate next work section
   still accurate?

3. **Process documents** (`AI.md`, `DEVELOPMENT.md`). Did any
   development rules prove unworkable, insufficient, or in need of
   precision?

4. **Surprises and pain points**. What was harder than expected? What
   design decisions caused rework? What would have been better to
   decide earlier? Capture these as updates to `ARCHITECTURE.md`,
   `DEVELOPMENT.md`, or `AI.md` — wherever the lesson is most
   actionable for the next epoch.

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

Before making or proposing a significant architectural change, run the
architecture stress-review checklist at
`pr-review/architecture-checklist.md`. Include the stress-review result
in the PR description or review report.

Any architecture review should also include a subjective pass over the
`*(Current inconsistency: ...)*` annotations in `ARCHITECTURE.md
§Architectural basis`. For each one, ask: has recent work resolved this
inconsistency? If yes, remove the annotation and close or update the
corresponding ROADMAP.md gap-closure item.

---

## Physics capability implementation paths

Every PR that adds or changes a *physics capability* is in one of
three lanes:

- **Lane A — Port-and-verify.** Adapt from a permissively-licensed
  reference code with attribution. Default for permissive references.
  No derivation document required.
- **Lane B — Clean-room from paper.** Mandatory when the only
  reference is copyleft-licensed (per
  [`docs/research/06-12-licensing.md`](docs/research/06-12-licensing.md):
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

Lanes B and C require a `_derive()` function in the production module
with executable SymPy checks on load-bearing algebraic steps. A paired
`generate()` function produces the runtime constants block; running
`scripts/generate_kernels.py` splices it back into the module.
Infrastructure capabilities (dispatch, mesh topology, I/O, field
placement) are out of scope.

The lane must be stated in the PR description for any physics
capability, e.g. `Lane C (origination). Reference papers: [...]`.
For Lane B, explicitly record that reference source was not consulted.
