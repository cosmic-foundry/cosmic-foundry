# Agent Instructions

These guidelines apply to all AI agents working on this repository,
regardless of platform (Claude Code, Codex, Gemini, or others).

## Development Rules

The authoritative source is
[ADR-0005](adr/ADR-0005-branch-pr-attribution-discipline.md). This
section is the informal quick-reference; when the two disagree, the
ADR wins.

### Branches and PRs

- Only work on a fork of the upstream repository.
- Every change lands via a pull request; no direct commits to
  `upstream/main`.
- Create topic branches from `origin/main` (the fork's main), not
  from `upstream/main` directly. Syncing `origin/main` to
  `upstream/main` is an explicit separate step.
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

## Architectural Decisions

Architectural decisions are recorded as ADRs in `adr/`. **At the
start of every session**, read `adr/README.md` — it indexes every
ADR in force. When work touches a topic listed there, read the
full ADR before making changes; the index is a pointer, not a
summary substitute.

When making a new architectural decision, copy
`adr/adr-template.md` to `adr/ADR-NNNN-<short-title>.md`, mark it
Proposed, and add a line to `adr/README.md` in the same PR.

Before treating an architectural decision as ready for human review,
run `pr-review/architecture-checklist.md`. The checklist forces an
agent or reviewer to map the design space, define concept ownership,
write realistic usage traces, normalize dependencies and lowering
boundaries, and identify fences / materialization points. Include the
stress-review result in the ADR, the PR description, or the review
report.

Accepted ADRs may be amended in place when a conversation implies
a change consistent with the existing decision — propose the edit
directly rather than routing clarifications through AI.md. Each
amendment appends a dated bullet to the ADR's *Amendments*
section. Reversing a decision still goes via supersession (a new
ADR). See
[ADR-0005 §Decision → ADR amendment policy](adr/ADR-0005-branch-pr-attribution-discipline.md#adr-amendment-policy).

## Code Quality

- Write code that is:
  - Self-documenting (clear variable names, simple logic)
  - Minimal (only what's needed, no over-engineering)
  - Testable (existing tests must pass, new features need tests)
