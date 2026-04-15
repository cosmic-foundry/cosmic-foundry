#!/bin/bash

set -euo pipefail

LAUNCH_DIR=$(pwd)

if [ -f "AI.md" ] && [ -d "scripts" ]; then
    ENGINE_ROOT=$LAUNCH_DIR
    DOC_PREFIX=""
    WORKSPACE_NOTE=""
elif [ -f "cosmic-foundry/AI.md" ] && [ -d "cosmic-foundry/scripts" ]; then
    ENGINE_ROOT="$LAUNCH_DIR/cosmic-foundry"
    DOC_PREFIX="cosmic-foundry/"
    WORKSPACE_NOTE="

This session was launched from the parent workspace:
- $LAUNCH_DIR

The sibling repositories available from this workspace are:"
    for repo in "$LAUNCH_DIR"/*; do
        if [ -d "$repo/.git" ]; then
            WORKSPACE_NOTE="$WORKSPACE_NOTE
- $repo"
        fi
    done
    WORKSPACE_NOTE="$WORKSPACE_NOTE

When work spans multiple repositories, use separate git worktrees and pull requests for each repository. Keep reusable engine changes in cosmic-foundry and application-layer changes in the relevant domain repository."
else
    echo "✗ Could not find Cosmic Foundry project docs."
    echo "Run from either:"
    echo "  - the cosmic-foundry repo root"
    echo "  - the parent directory containing ./cosmic-foundry"
    exit 1
fi

# Activate the repo-provided conda environment. This makes it available to the
# agent process and all subprocesses it spawns (Bash tool calls, pre-commit
# hooks, pytest, mypy, sphinx-build, etc.). activate_environment.sh checks for
# miniforge and exits with a clear message if it is missing.
source "$ENGINE_ROOT/environment/activate_environment.sh"

# Auto-sync when environment spec files have changed since the last
# setup/update run. setup_environment.sh is idempotent — it skips the
# Miniforge download if already present and handles both create and update —
# so calling it here is safe. The sentinel it writes is used to avoid
# re-running on every agent start.
_SENTINEL="${ENGINE_ROOT}/miniforge/envs/cosmic_foundry/.env_last_updated"
_needs_update=0
if [ ! -f "$_SENTINEL" ]; then
    _needs_update=1
elif [ "${ENGINE_ROOT}/environment/cosmic_foundry.yml" -nt "$_SENTINEL" ] \
  || [ "${ENGINE_ROOT}/environment/setup_environment.sh" -nt "$_SENTINEL" ]; then
    _needs_update=1
fi
if [ "$_needs_update" = "1" ]; then
    echo "Environment spec changed — re-running setup..."
    bash "$ENGINE_ROOT/environment/setup_environment.sh"
fi
unset _SENTINEL _needs_update

AGENT_TYPE=$1

case $AGENT_TYPE in
    gemini)
        CMD="gemini"
        DOC="GEMINI.md"
        ;;
    claude)
        CMD="claude"
        DOC="CLAUDE.md"
        ;;
    codex)
        CMD="codex"
        DOC="CODEX.md"
        ;;
    *)
        echo "Usage: ./scripts/start_agent.sh [gemini|claude|codex]"
        echo "Or from a parent workspace: ./cosmic-foundry/scripts/start_agent.sh [gemini|claude|codex]"
        exit 1
        ;;
esac

# initialization prompt
INIT_PROMPT="I am starting a new session from this workspace root:
$LAUNCH_DIR

The cosmic-foundry repository is at:
$ENGINE_ROOT

Please initialize your context by reading the following documents in order to understand the project rules, tools, and coordination patterns:
1. $ENGINE_ROOT/AI.md
2. $ENGINE_ROOT/$DOC

Use the repository path above for cosmic-foundry work. If work spans sibling repositories, first locate and read each repository's local agent instructions before changing it.$WORKSPACE_NOTE"

# Start the selected agent with the prompt
echo "Starting $AGENT_TYPE..."
$CMD "$INIT_PROMPT"
