#!/bin/bash
# Wrapper that enforces session timeout based on per-suite budget constants.
# Computes timeout = (solver_budget + integrator_budget + fixed_overhead) × tolerance
# (single source of truth from tests/claims.py)

set -e

TIMEOUT=$(python -c "
from tests.claims import (
    SOLVER_CONVERGENCE_BUDGET_S,
    INTEGRATOR_SESSION_BUDGET_S,
    FIXED_SESSION_OVERHEAD_S,
    BUDGET_TOLERANCE,
)
timeout_s = (
    SOLVER_CONVERGENCE_BUDGET_S + INTEGRATOR_SESSION_BUDGET_S + FIXED_SESSION_OVERHEAD_S
) * BUDGET_TOLERANCE
print(f'{timeout_s:.1f}')
")

echo "Running pytest with timeout: ${TIMEOUT}s"
timeout "$TIMEOUT" python -m pytest "$@"
