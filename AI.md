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
- CI's `pre-commit` job is a required status check on
  `upstream/main`; PRs cannot merge red.

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

## Environment

This repo uses miniforge to provide a self-contained environment.
DO NOT use system Python or any external `python`/`pytest` command.

### Setup (One-Time)

If `miniforge/` directory is missing:
```bash
bash environment/setup_environment.sh
```

### Before Any Work

**Every session**, verify miniforge exists:
```bash
test -d miniforge && echo "✓ miniforge found" || echo "✗ Run setup script"
```

If the session starts from a parent organization workspace, run the same check
against the `cosmic-foundry` checkout path:
```bash
test -d cosmic-foundry/miniforge && echo "✓ miniforge found" || echo "✗ Run setup script"
```

If missing, ask user to run the setup script before proceeding.

## Architectural Decisions

Architectural decisions are recorded as ADRs in `adr/`. **At the
start of every session**, read `adr/README.md` — it indexes every
ADR in force. When work touches a topic listed there, read the
full ADR before making changes; the index is a pointer, not a
summary substitute.

When making a new architectural decision, copy
`adr/adr-template.md` to `adr/ADR-NNNN-<short-title>.md`, mark it
Proposed, and add a line to `adr/README.md` in the same PR.

## Code Quality

- Write code that is:
  - Self-documenting (clear variable names, simple logic)
  - Minimal (only what's needed, no over-engineering)
  - Testable (existing tests must pass, new features need tests)
