#!/bin/bash
# Run the repository's adversarial PR-review prompt through Codex.
#
# Invocation glue for pr-review/. Fetches PR metadata and diff with gh,
# then passes that context plus the tracked reviewer spec to a read-only
# Codex session. Set COSMIC_FOUNDRY_PR_REPO to override the default
# upstream repository (cosmic-foundry/cosmic-foundry).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# shellcheck source=scripts/_review_pr_impl.sh
source "$SCRIPT_DIR/_review_pr_impl.sh" "$@"

codex exec --sandbox read-only - <<< "$REVIEW_PROMPT"
