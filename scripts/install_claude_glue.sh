#!/bin/bash
# Generate the Claude Code invocation glue for this repository.
#
# Writes files into `.claude/` (gitignored). The glue is a thin
# pointer to the in-repo reviewer spec under `pr-review/` — that
# directory is the single source of truth. Tool-specific glue stays
# out of the repo so the project artifact can evolve independently
# of any particular AI tool.
#
# Idempotent: safe to rerun. Overwrites generated files each
# invocation so updates propagate on the next setup.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
CLAUDE_DIR="${REPO_ROOT}/.claude"
COMMANDS_DIR="${CLAUDE_DIR}/commands"
AGENTS_DIR="${CLAUDE_DIR}/agents"

mkdir -p "$COMMANDS_DIR" "$AGENTS_DIR"

cat >"${COMMANDS_DIR}/review-pr.md" <<'EOF'
---
description: Run the adversarial PR reviewer against a pull request.
argument-hint: <pr-number>
---

<!-- GENERATED FILE — edits will be lost on the next
     setup_environment.sh run. Source of truth:
     pr-review/ (tracked) + scripts/install_claude_glue.sh. -->

You are orchestrating the project's adversarial PR review. The
reviewer's role, protocol, and output format are defined in
`pr-review/agent.md`. The checklist of historical failure modes
is in `pr-review/checklist.md`.

Perform these steps:

1. If `$ARGUMENTS` is empty, ask the user for the PR number
   before proceeding.
2. Fetch PR metadata and diff:
   ```bash
   gh pr view $ARGUMENTS --repo cosmic-foundry/cosmic-foundry
   gh pr diff $ARGUMENTS --repo cosmic-foundry/cosmic-foundry
   ```
3. Spawn the `pr-reviewer` subagent (Sonnet). Pass the PR
   number, the fetched metadata, and the diff as context.
   The subagent uses only Read/Grep/Glob for working-tree
   inspection — do not ask it to run shell commands.
4. Return the subagent's report verbatim. Do not summarize,
   soften, or re-order findings. Surface the "Confidence and
   blind spots" section prominently.

If the subagent returns a malformed output, say so explicitly
rather than papering over it.
EOF

cat >"${AGENTS_DIR}/pr-reviewer.md" <<'EOF'
---
name: pr-reviewer
description: Adversarial PR reviewer for this repository. Walks the checklist of historical failure modes and produces a structured report. Use via the /review-pr slash command.
model: sonnet
tools: Read, Grep, Glob
---

<!-- GENERATED FILE — edits will be lost on the next
     setup_environment.sh run. Source of truth: pr-review/. -->

Your role, protocol, and output format are defined in
`pr-review/agent.md`. The catalog of historical failure modes
you must walk on every review is in `pr-review/checklist.md`.

On invocation:

1. Read `pr-review/agent.md` in full.
2. Read `pr-review/checklist.md` in full.
3. Follow the protocol in `pr-review/agent.md` using the PR
   metadata and diff passed to you by the invoking command.
   Use Read, Grep, and Glob for working-tree inspection only —
   all external PR data is already provided as text.
4. Produce the report in the exact format specified in
   `pr-review/agent.md`.

Do not modify the working tree, push, comment on the PR, or
take any action visible outside this review.
EOF

echo "Installed Claude Code glue:"
echo "  ${COMMANDS_DIR}/review-pr.md"
echo "  ${AGENTS_DIR}/pr-reviewer.md  (Sonnet)"
