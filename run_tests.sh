#!/bin/bash
# Wrapper that enforces session timeout based on COSMIC_FOUNDRY_TEST_BUDGET_S.
# Computes timeout = (budget + fixed_overhead) × tolerance
# (single source of truth from tests/claims.py)

set -e

TIMEOUT=$(python -c "
from tests.claims import MAX_WALLTIME_S, FIXED_SESSION_OVERHEAD_S, BUDGET_TOLERANCE
timeout_s = (MAX_WALLTIME_S + FIXED_SESSION_OVERHEAD_S) * BUDGET_TOLERANCE
print(f'{timeout_s:.1f}')
")

echo "Running pytest with timeout: ${TIMEOUT}s"
timeout "$TIMEOUT" python -m pytest "$@"
