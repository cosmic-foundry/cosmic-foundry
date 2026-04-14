#!/bin/bash
# Source this script to activate the cosmic_foundry conda environment:
#
#   source environment/activate_environment.sh
#
# It must be sourced, not executed, because conda activation modifies
# the current shell's PATH and environment variables.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "Error: this script must be sourced, not executed." >&2
  echo "       Run: source ${BASH_SOURCE[0]}" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
MINIFORGE_DIR="${REPO_ROOT}/miniforge"
CONDA_SH="${MINIFORGE_DIR}/etc/profile.d/conda.sh"

if [ ! -f "$CONDA_SH" ]; then
  echo "Error: miniforge not found at $MINIFORGE_DIR." >&2
  echo "       Run environment/setup_environment.sh first." >&2
  return 1
fi

source "$CONDA_SH"
conda activate cosmic_foundry
