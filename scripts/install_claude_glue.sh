#!/bin/bash
# Generate the Claude Code invocation glue for this repository.
#
# This is a setup-time generator. It writes files into `.claude/`,
# which is gitignored. The glue is a thin pointer to the in-repo
# reviewer spec under `pr-review/` — that directory is the single
# source of truth for the reviewer's role and checklist. Tool-specific
# glue stays out of the repo so the project artifact can evolve
# independently of any particular AI tool.
#
# Idempotent: safe to rerun. Overwrites the generated files each
# invocation so updates to this script propagate on the next setup.

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

You are invoking the project's adversarial PR reviewer. The
reviewer's role, protocol, and output format are defined in
`pr-review/agent.md`. The checklist of historical failure modes
is in `pr-review/checklist.md`. Both live in the repo so they
evolve with the project; the slash command you are reading is
generated glue (see `scripts/install_claude_glue.sh`).

Perform these steps:

1. Read `pr-review/agent.md` and `pr-review/checklist.md` in
   full. Do not skim.
2. Fetch the PR metadata and diff:
   ```bash
   gh pr view $ARGUMENTS --repo cosmic-foundry/cosmic-foundry
   gh pr diff $ARGUMENTS --repo cosmic-foundry/cosmic-foundry
   ```
   If `$ARGUMENTS` is empty, ask the user for the PR number
   before proceeding.
3. Spawn the `pr-reviewer` subagent with the PR number, the
   fetched metadata, and the diff as context. The subagent
   will follow the protocol defined in `pr-review/agent.md`
   and produce the report in the format specified there.
4. Return the subagent's report to the user verbatim. Do not
   summarise, soften, or re-order the findings.

The reviewer is same-model (Claude reviewing Claude). Surface
the "Confidence and blind spots" section of the report
prominently — it is the reviewer's own statement of where this
review is weakest.
EOF

cat >"${AGENTS_DIR}/pr-reviewer.md" <<'EOF'
---
name: pr-reviewer
description: Adversarial PR reviewer specialised for this repository's historical failure modes. Use via the /review-pr slash command.
tools: Read, Grep, Glob, Bash
---

Your role, protocol, and output format are defined in
`pr-review/agent.md` at the repository root. The catalogue of
historical failure modes you must walk on every review is in
`pr-review/checklist.md`.

On invocation:

1. Read `pr-review/agent.md` in full. That document overrides
   anything you may have inferred from this stub.
2. Read `pr-review/checklist.md` in full.
3. Follow the protocol in `pr-review/agent.md`, using the PR
   number and diff passed to you by the invoking command.
4. Produce the report in the exact format specified in
   `pr-review/agent.md`.

Do not modify the working tree, push, comment on the PR, or
take any action visible outside this review. Output is text
only.
EOF

echo "Installed Claude Code glue:"
echo "  ${COMMANDS_DIR}/review-pr.md"
echo "  ${AGENTS_DIR}/pr-reviewer.md"
