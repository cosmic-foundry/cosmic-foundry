#!/bin/bash
# Run the repository's adversarial PR-review prompt through Codex.
#
# This is the Codex invocation glue for pr-review/. It mirrors the
# Claude Code /review-pr command by pre-fetching PR metadata and diff
# with gh, then passing that text plus the tracked reviewer spec to a
# read-only Codex session.

set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: ./scripts/review_pr_with_codex.sh <pr-number>"
    exit 2
fi

PR_NUMBER="$1"
REPO="${COSMIC_FOUNDRY_PR_REPO:-cosmic-foundry/cosmic-foundry}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$REPO_ROOT"

PR_METADATA="$(gh pr view "$PR_NUMBER" --repo "$REPO")"
PR_DIFF="$(gh pr diff "$PR_NUMBER" --repo "$REPO")"

codex exec --sandbox read-only - <<EOF
You are running Cosmic Foundry's adversarial PR reviewer for PR
#${PR_NUMBER} in ${REPO}.

The reviewer's role, protocol, checklist, and output format are
tracked in this repository:

- pr-review/agent.md
- pr-review/checklist.md

Perform these steps:

1. Read pr-review/agent.md in full.
2. Read pr-review/checklist.md in full.
3. Review the PR using the metadata and diff below as the external
   PR context. Use the working tree only for read-only inspection.
4. Return the report in the exact format specified by
   pr-review/agent.md.

Do not modify the working tree, push, comment on the PR, or take any
action visible outside this review.

## PR metadata

${PR_METADATA}

## PR diff

${PR_DIFF}
EOF
