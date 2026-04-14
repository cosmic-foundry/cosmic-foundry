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

# check for miniforge
if [ ! -d "$ENGINE_ROOT/miniforge" ]; then
    echo "✗ miniforge not found at $ENGINE_ROOT/miniforge."
    echo "Run 'bash $DOC_PREFIX""environment/setup_environment.sh' from the launch directory, or run setup from $ENGINE_ROOT."
    exit 1
fi

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
