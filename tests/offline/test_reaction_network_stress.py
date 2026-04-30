"""Offline stress tests for large-scale parametric reaction-network NSE integration.

These tests are intentionally large and opt-in.  They exercise the
ConstraintAwareController and solve_nse path on networks too large for CI:
longer chains and larger spoke networks than the CI sequences.

Expand _CHAIN_N_OFFLINE and _SPOKE_N_OFFLINE once performance improves.

Run explicitly with:

    COSMIC_FOUNDRY_OFFLINE_NETWORK_STRESS=1 pytest tests/offline
"""

from __future__ import annotations

import os

import pytest

from tests.parametric_networks import (
    _ParametricNSEClaim,
    chain_claims,
    spoke_claims,
)

_RUN_OFFLINE = os.environ.get("COSMIC_FOUNDRY_OFFLINE_NETWORK_STRESS") == "1"

pytestmark = [
    pytest.mark.offline,
    pytest.mark.skipif(
        not _RUN_OFFLINE,
        reason=(
            "offline network stress tests; set "
            "COSMIC_FOUNDRY_OFFLINE_NETWORK_STRESS=1 to run"
        ),
    ),
]

# Expand these as performance improves.
_CHAIN_N_OFFLINE = range(5, 12)  # n_species = 5 .. 11
_SPOKE_N_OFFLINE = range(7, 22)  # n_species = 7 .. 21
_SPOKE_K_OFFLINE = [1, 10, 100]  # k_fast/k_slow ratios

_OFFLINE_CLAIMS: list[_ParametricNSEClaim] = [
    *chain_claims(_CHAIN_N_OFFLINE),
    *spoke_claims(_SPOKE_N_OFFLINE, _SPOKE_K_OFFLINE),
]


@pytest.mark.parametrize(
    "claim",
    _OFFLINE_CLAIMS,
    ids=[c.description for c in _OFFLINE_CLAIMS],
)
def test_offline_network_nse(claim: _ParametricNSEClaim) -> None:
    claim.check()
