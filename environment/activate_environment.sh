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

# Conda's toolchain activate.d / deactivate.d scripts (e.g. cctools on
# macOS arm64) reference CONDA_BACKUP_* variables unconditionally, which
# aborts a caller running under `set -u`. Relax nounset across the
# activation, then restore the caller's prior state so we don't silently
# weaken their shell hardening. See https://github.com/conda/conda/issues/8186.
case "$-" in
  *u*) _prev_nounset=1 ;;
  *)   _prev_nounset=0 ;;
esac
set +u
source "$CONDA_SH"
conda activate cosmic_foundry
[ "$_prev_nounset" = 1 ] && set -u
unset _prev_nounset
