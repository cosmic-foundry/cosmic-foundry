# PR review — adversarial reviewer

This directory is the single source of truth for the project's
adversarial PR-review capability. It contains the **what** of
review — the reviewer's role, protocol, and the checklist of
historical failure modes to look for — independent of any
particular AI tool.

Tool-specific invocation glue (Claude Code slash commands, Codex
prompts, Gemini configs) is generated at environment-setup time
from these files and is **not** tracked in the repository. See
`scripts/install_claude_glue.sh` and the call site in
`environment/setup_environment.sh`.

## Files

- [`agent.md`](agent.md) — reviewer role, inputs, protocol, and
  output format. Read this first.
- [`checklist.md`](checklist.md) — catalog of historical
  failure modes on this repository, grouped by theme. The
  reviewer is expected to walk this on every PR.

## How a reviewer is invoked

The reviewer is a same-model subagent (Claude reviewing Claude's
work). That is a known limitation: shared blind spots between
author and reviewer are expected. Cross-model review is a
follow-up; when it lands, the glue layer gains a second install
script that points at the same `agent.md` and `checklist.md`
here.

Under Claude Code, after running `environment/setup_environment.sh`
(which writes `.claude/commands/review-pr.md` and
`.claude/agents/pr-reviewer.md`), invoke:

```text
/review-pr <pr-number>
```

The command fetches the PR metadata and diff via `gh`, then
spawns the `pr-reviewer` subagent with `agent.md` and
`checklist.md` in context.

## Updating

Treat the checklist as a living document. When a new failure
mode escapes review, add it as a new entry under the appropriate
theme (or as a new theme). When an entry becomes obsolete —
because the failure mode is now caught by CI, a linter, or a
pre-commit hook — delete it, don't leave a stale line item.
