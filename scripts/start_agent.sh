#!/bin/bash

set -euo pipefail

LAUNCH_DIR=$(pwd -P)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
ENGINE_ROOT=$(cd "$SCRIPT_DIR/.." && pwd -P)
ENGINE_PARENT=$(dirname "$ENGINE_ROOT")

if [ ! -f "$ENGINE_ROOT/DEVELOPMENT.md" ] || [ ! -d "$ENGINE_ROOT/scripts" ]; then
    echo "✗ Could not find Cosmic Foundry project docs next to this launcher."
    exit 1
fi

workspace_note() {
    local workspace=$1
    local note="

The sibling repositories available from this workspace are:"
    for repo in "$workspace"/*; do
        if [ -d "$repo/.git" ]; then
            note="$note
- $repo"
        fi
    done
    note="$note

When work spans multiple repositories, use separate git worktrees and pull requests for each repository. Keep reusable engine changes in cosmic-foundry and application-layer changes in the relevant domain repository."
    printf '%s' "$note"
}

if [ "$LAUNCH_DIR" = "$ENGINE_ROOT" ]; then
    WORKSPACE_NOTE=""
elif [ "$LAUNCH_DIR" = "$ENGINE_PARENT" ]; then
    WORKSPACE_NOTE="

This session was launched from the parent workspace:
- $LAUNCH_DIR$(workspace_note "$LAUNCH_DIR")"
elif [ "$(dirname "$LAUNCH_DIR")" = "$ENGINE_PARENT" ] && [ -d "$LAUNCH_DIR/.git" ]; then
    WORKSPACE_NOTE="

This session was launched from a sibling repository:
- $LAUNCH_DIR

Treat this sibling repository as the primary working repository for this session. Before changing it, read its local agent instructions if present:
- $LAUNCH_DIR/DEVELOPMENT.md
- any other agent instruction document in $LAUNCH_DIR

The parent workspace is:
- $ENGINE_PARENT$(workspace_note "$ENGINE_PARENT")"
else
    echo "✗ Could not find Cosmic Foundry project docs."
    echo "Run from either:"
    echo "  - the cosmic-foundry repo root"
    echo "  - the parent directory containing ./cosmic-foundry"
    echo "  - a sibling repository next to cosmic-foundry"
    exit 1
fi

# Activate the repo-provided conda environment. This makes it available to the
# agent process and all subprocesses it spawns (Bash tool calls, pre-commit
# hooks, pytest, mypy, sphinx-build, etc.). activate_environment.sh checks for
# miniforge and exits with a clear message if it is missing.
source "$ENGINE_ROOT/scripts/activate_environment.sh"

# Auto-sync when environment spec files have changed since the last
# setup/update run. setup_environment.sh is idempotent — it skips the
# Miniforge download if already present and handles both create and update —
# so calling it here is safe. The sentinel it writes is used to avoid
# re-running on every agent start.
_SENTINEL="${ENGINE_ROOT}/environment/miniforge/envs/cosmic_foundry/.env_last_updated"
_needs_update=0
if [ ! -f "$_SENTINEL" ]; then
    _needs_update=1
elif [ "${ENGINE_ROOT}/environment/cosmic_foundry.yml" -nt "$_SENTINEL" ] \
  || [ "${ENGINE_ROOT}/scripts/setup_environment.sh" -nt "$_SENTINEL" ]; then
    _needs_update=1
fi
if [ "$_needs_update" = "1" ]; then
    echo "Environment spec changed — re-running setup..."
    bash "$ENGINE_ROOT/scripts/setup_environment.sh"
fi
unset _SENTINEL _needs_update

AGENT_TYPE=${1:-}

case $AGENT_TYPE in
    gemini)
        CMD="gemini"
        ;;
    claude)
        CMD="claude"
        ;;
    codex)
        CMD="codex"
        ;;
    *)
        echo "Usage: ./scripts/start_agent.sh [gemini|claude|codex]"
        echo "Or from a parent workspace: ./cosmic-foundry/scripts/start_agent.sh [gemini|claude|codex]"
        echo "Or from a sibling repo: ../cosmic-foundry/scripts/start_agent.sh [gemini|claude|codex]"
        exit 1
        ;;
esac

# initialization prompt
INIT_PROMPT="I am starting a new session from this workspace root:
$LAUNCH_DIR

The cosmic-foundry repository is at:
$ENGINE_ROOT

Please initialize your context by reading the following document in order to understand the project rules, tools, and coordination patterns:
1. $ENGINE_ROOT/DEVELOPMENT.md

Use the repository path above for cosmic-foundry work. If work spans sibling repositories, first locate and read each repository's local agent instructions before changing it.$WORKSPACE_NOTE"

# Start the selected agent with the prompt
echo "Starting $AGENT_TYPE..."
$CMD "$INIT_PROMPT"
