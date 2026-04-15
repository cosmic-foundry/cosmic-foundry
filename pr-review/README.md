# PR review — adversarial reviewer

This directory is the single source of truth for the project's
adversarial PR-review capability. It contains the **what** of
review — the reviewer's role, protocol, and the checklist of
historical failure modes — independent of any particular AI
tool.

Tool-specific invocation glue is a thin layer over these files.
Generated glue is installed at environment-setup time and is **not**
tracked in the repository. Tracked wrapper scripts live under
`scripts/` when a tool has no generated project-local command
format. See `scripts/install_claude_glue.sh`,
`scripts/review_pr_with_codex.sh`, and the call site in
`environment/setup_environment.sh`.

## Files

- [`agent.md`](agent.md) — reviewer role, inputs, protocol,
  and output format. Read this first.
- [`checklist.md`](checklist.md) — catalog of historical
  failure modes on this repository, grouped by theme. The
  reviewer walks this on every PR.

## Architecture

The reviewer spec is tool-independent. Each agent has a thin wrapper
script under `scripts/` that pre-fetches all external data
(`gh pr view`, `gh pr diff`) and passes it as text; the reviewer uses
the working tree only for read-only inspection, so no shell-command
approvals are needed during review. The shared fetch logic and prompt
live in `scripts/_review_pr_impl.sh`; the per-agent scripts source it
and supply the single agent-specific CLI invocation.

Upstream enforces squash merge, so the PR diff against the target
branch is the unit of review — individual commit history is not
examined.

Mechanical pattern-matching checks belong in `scripts/ci/` and
`.pre-commit-config.yaml`, not in this reviewer. The checklist
covers judgment-heavy items that CI cannot catch, plus a few
items that are not yet automated (marked *pending pre-commit*).

Reviewer blind spots depend on the tool/model doing the review. A
same-tool review is a known limitation when the author used the same
agent family; cross-tool review is preferred for higher-risk PRs.

## Invoking

From a terminal or automation:

```bash
./scripts/review_pr_with_claude.sh <pr-number>   # Claude Code
./scripts/review_pr_with_codex.sh <pr-number>    # Codex
./scripts/review_pr_with_gemini.sh <pr-number>   # Gemini CLI
```

Set `COSMIC_FOUNDRY_PR_REPO` to override the default upstream
repository (`cosmic-foundry/cosmic-foundry`).

Inside an active session (any agent), either of these user requests
means "run the adversarial PR reviewer for that pull request":

```text
Review PR <pr-number>
Review <pr-number>
```

The agent should read `pr-review/agent.md` and `pr-review/checklist.md`,
fetch PR metadata and diff with
`gh pr view <pr-number> --repo cosmic-foundry/cosmic-foundry` and
`gh pr diff <pr-number> --repo cosmic-foundry/cosmic-foundry`, then
return the exact report format required by `pr-review/agent.md`.

## Updating the checklist

Add entries when a failure escapes review. Delete entries when
a pre-commit script now catches them — don't leave stale items.
Items marked *pending pre-commit* should be moved to
`scripts/ci/` and wired to the hook when a contributor gets
around to scripting them.
