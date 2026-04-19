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

Implementation effort is a tiebreaker at most. Present options in
terms of these downstream costs, not in terms of "cheap vs. expensive
to build." Always include the lower-complexity option in the
comparison even when recommending a richer one — the goal is to keep
the decision with the user, not to bias toward ambition just because
ambition is cheap to type.

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

---

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

**If the env check fails** (script prints `✗ WRONG ENVIRONMENT` and
exits non-zero), stop immediately. The correct way to start a session
is:
```bash
./scripts/start_agent.sh claude   # or gemini / codex
```
`start_agent.sh` activates the environment automatically before
launching the agent. Alternatively, activate manually:
```bash
source environment/activate_environment.sh
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
root. It is the navigation anchor: the directory map, planned modules
not yet coded, and the immediate next work. Read `ROADMAP.md` for the
high-level capability sequence across both the simulation and V&V
tracks.

Every PR should state whether it advances the simulation track, the
V&V track, or both. Cross-track dependencies must be explicit in the
PR description.

Maintenance and tooling PRs that do not advance either track should
note "No change to roadmap position" in the PR description.

---

## Implementation plans

The immediate next work for both tracks is maintained in the
`## Immediate next work` section of `ROADMAP.md`. Keep it current:
when a sprint item completes, remove it; when new immediate work
becomes clear, add it. Do not plan beyond what is concretely
unblocked — if an item depends on decisions not yet made, it does not
belong in the immediate section.

---

## Epoch retrospective

**When an epoch is declared complete**, before opening any code PR for
the next epoch, perform a retrospective review. The retrospective
produces only documentation changes — ADR edits, roadmap updates,
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

3. **`replication/` and formulas register**. Do the Function: blocks on
   implemented operator classes reflect what was actually built? Are
   there formula entries that should be updated now rather than left
   stale?

4. **Process documents** (`AI.md`, `DEVELOPMENT.md`). Did any
   development rules prove unworkable, insufficient, or in need of
   precision?

5. **Surprises and pain points**. What was harder than expected? What
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

---

## Physics capability implementation paths

Every PR that adds or changes a *physics capability* is in one of
three lanes:

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

The lane must be stated in the PR description for any physics
capability, e.g. `Lane C (origination). Reference papers: [...]`.
For Lane B, explicitly record that reference source was not consulted.
