#!/bin/bash
# Wrapper that enforces session timeout based on COSMIC_FOUNDRY_TEST_BUDGET_S.
# Computes timeout = budget × tolerance (single source of truth from tests/claims.py)

set -e

# Extract MAX_WALLTIME_S and BUDGET_TOLERANCE from tests/claims.py
BUDGET=$(python -c "
from tests.claims import MAX_WALLTIME_S, BUDGET_TOLERANCE
timeout_s = MAX_WALLTIME_S * BUDGET_TOLERANCE
print(f'{timeout_s:.1f}')
")

echo "Running pytest with timeout: ${BUDGET}s"
timeout "$BUDGET" python -m pytest "$@"
