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

- [`agent.md`](agent.md) — reviewer roles (sweep + synthesis),
  inputs, protocols, and output formats. Read this first.
- [`checklist.md`](checklist.md) — catalog of historical
  failure modes on this repository, grouped by theme and
  tagged `[sweep]` or `[synthesis]` per item. The reviewer
  walks this on every PR.

## How a reviewer is invoked

The review runs in two tiers so each part uses a model sized
to the work: a **sweep** pass on Haiku walks the mechanical
`[sweep]` items in the checklist and produces structured
findings, and a **synthesis** pass on Sonnet walks the
judgment-heavy `[synthesis]` items, reads the diff end-to-end,
and assembles the final report using the sweep's output as
input.

Both passes are same-model (Claude reviewing Claude) — a
known limitation; cross-model review is a follow-up that adds
a parallel `install_<tool>_glue.sh` pointing at the same
in-repo spec without duplicating it.

Under Claude Code, after running `environment/setup_environment.sh`
(which writes `.claude/commands/review-pr.md`,
`.claude/agents/pr-reviewer-sweep.md`, and
`.claude/agents/pr-reviewer.md`), invoke:

```text
/review-pr <pr-number>
```

The command fetches PR metadata, commits, and diff via `gh`,
spawns `pr-reviewer-sweep` (Haiku) to produce the sweep
findings, then spawns `pr-reviewer` (Sonnet) with the PR plus
the sweep findings and returns its report.

## Updating

Treat the checklist as a living document. When a new failure
mode escapes review, add it as a new entry under the appropriate
theme (or as a new theme). When an entry becomes obsolete —
because the failure mode is now caught by CI, a linter, or a
pre-commit hook — delete it, don't leave a stale line item.
