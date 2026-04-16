#!/bin/bash
# Shared implementation for review_pr_with_*.sh scripts.
# Source this from a wrapper — do not invoke directly.
#
# Expects the caller to have already set:
#   REPO_ROOT  — root of the repository
#
# Call as: source _review_pr_impl.sh "$@"
#
# On return, REVIEW_PROMPT is set to the full prompt text ready
# to pipe into the agent CLI.

if [ "$#" -ne 1 ]; then
    echo "Usage: $(basename "${BASH_SOURCE[1]:-$0}") <pr-number>" >&2
    return 2 2>/dev/null || exit 2
fi

PR_NUMBER="$1"
REPO="${COSMIC_FOUNDRY_PR_REPO:-cosmic-foundry/cosmic-foundry}"

cd "$REPO_ROOT"

echo "Fetching PR #${PR_NUMBER} from ${REPO}..." >&2
PR_METADATA="$(gh pr view "$PR_NUMBER" --repo "$REPO")"
PR_DIFF="$(gh pr diff "$PR_NUMBER" --repo "$REPO")"

REVIEW_PROMPT="You are running Cosmic Foundry's adversarial PR reviewer for PR
#${PR_NUMBER} in ${REPO}.

The reviewer's role, protocol, checklists, and output format are
tracked in this repository:

- pr-review/agent.md
- pr-review/checklist.md
- pr-review/architecture-checklist.md

Perform these steps:

1. Read pr-review/agent.md in full.
2. Read pr-review/checklist.md in full.
3. If the PR is architecture-changing, also read
   pr-review/architecture-checklist.md and run its stress-review
   protocol.
4. Review the PR using the metadata and diff below as the external
   PR context. Use the working tree only for read-only inspection.
5. Return the report in the exact format specified by
   pr-review/agent.md.

Do not modify the working tree, push, comment on the PR, or take any
action visible outside this review.

## PR metadata

${PR_METADATA}

## PR diff

${PR_DIFF}"
