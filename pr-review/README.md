# PR review — adversarial reviewer

This directory is the single source of truth for the project's
adversarial PR-review capability. It contains the **what** of
review — the reviewer's role, protocol, and the checklist of
historical failure modes — independent of any particular AI
tool.

Tool-specific invocation glue (Claude Code slash commands, Codex
prompts, Gemini configs) is generated at environment-setup time
from these files and is **not** tracked in the repository. See
`scripts/install_claude_glue.sh` and the call site in
`environment/setup_environment.sh`.

## Files

- [`agent.md`](agent.md) — reviewer role, inputs, protocol,
  and output format. Read this first.
- [`checklist.md`](checklist.md) — catalog of historical
  failure modes on this repository, grouped by theme. The
  reviewer walks this on every PR.

## Architecture

The reviewer is a single Sonnet subagent. The invoking `/review-pr`
command pre-fetches all external data (`gh pr view`, `gh pr diff`)
and passes it as text; the subagent uses only `Read`/`Grep`/`Glob`
for working-tree inspection, so no shell-command approvals are
needed during review.

Upstream enforces squash merge, so the PR diff against the target
branch is the unit of review — individual commit history is not
examined.

Mechanical pattern-matching checks belong in `scripts/ci/` and
`.pre-commit-config.yaml`, not in this reviewer. The checklist
covers judgment-heavy items that CI cannot catch, plus a few
items that are not yet automated (marked *pending pre-commit*).

The reviewer is same-model (Claude reviewing Claude) — a known
limitation. Cross-model review is a follow-up that adds a parallel
`install_<tool>_glue.sh` pointing at the same spec here.

## Invoking

After `environment/setup_environment.sh` has run:

```text
/review-pr <pr-number>
```

## Updating the checklist

Add entries when a failure escapes review. Delete entries when
a pre-commit script now catches them — don't leave stale items.
Items marked *pending pre-commit* should be moved to
`scripts/ci/` and wired to the hook when a contributor gets
around to scripting them.
