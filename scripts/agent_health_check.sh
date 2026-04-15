#!/usr/bin/env bash
# Startup health checks for agent sessions.
#
# Contract (see AI.md "Before Any Work"):
#   1. cosmic_foundry conda env is active
#   2. pre_commit Python package is importable
#   3. .git/hooks/pre-commit is installed
#
# Exit 1 on env-check failure (session-blocking); exit 0 otherwise.
#
# This script is the exact trust surface behind the
# `Bash(./scripts/agent_health_check.sh)` allow rule in
# .claude/settings.json. Keep it read-only: inspect state, never mutate.
set -u

if [[ "${CONDA_DEFAULT_ENV:-}" == "cosmic_foundry" ]]; then
  echo "✓ cosmic_foundry env active"
else
  echo "✗ WRONG ENVIRONMENT"
  exit 1
fi

if python -c "import pre_commit" 2>/dev/null; then
  echo "✓ pre-commit available"
else
  echo "✗ env stale — run 'conda env update -f environment/cosmic_foundry.yml --prune'"
fi

if test -x .git/hooks/pre-commit; then
  echo "✓ pre-commit git hook installed"
else
  echo "✗ run 'pre-commit install' inside the activated env"
fi
