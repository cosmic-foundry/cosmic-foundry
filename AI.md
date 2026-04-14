# Agent Instructions

These guidelines apply to all AI agents working on this repository,
regardless of platform (Claude Code, Codex, Gemini, or others).

## Development Rules

- Every commit should be no more than approximately 100 lines of code changes (larger diffs are fine to documentation).
- Only work on a fork of the upstream repository.
- Every change should be done on a branch and handled via a pull request on GitHub.
- Never force push.
- Never alter the git history.
- Never include the user's local absolute filesystem paths in commit messages,
  PR titles, PR descriptions, or other durable repository metadata. Use
  repository-relative paths or generic tool commands so private workstation
  details are not published.
- This project has not started versioning or published stable APIs yet. Do not
  preserve backwards compatibility by default during structural refactors.

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

## Code Quality

- Write code that is:
  - Self-documenting (clear variable names, simple logic)
  - Minimal (only what's needed, no over-engineering)
  - Testable (existing tests must pass, new features need tests)
