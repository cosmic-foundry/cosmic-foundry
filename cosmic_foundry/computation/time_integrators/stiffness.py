"""Stiffness diagnostics and Adams/BDF family switching for Nordsieck states."""

from __future__ import annotations

from typing import NamedTuple

from cosmic_foundry.computation.tensor import Tensor


class FamilyName(str):
    """Nordsieck corrector family name."""

    def __new__(cls, value: str) -> FamilyName:
        if value not in ("adams", "bdf"):
            raise ValueError("family must be 'adams' or 'bdf'")
        return str.__new__(cls, value)


class FamilySwitch(NamedTuple):
    """Family-switch decision for one accepted Nordsieck state."""

    family: FamilyName
    stiffness: float
    switched: bool


class StiffnessDiagnostic:
    """Streaming Gershgorin stiffness estimate from ``hJ``.

    For a dense Jacobian ``J``, the Gershgorin row-sum bound
    ``max_i Σ_j |h J_ij|`` upper-bounds the spectral radius of ``hJ``.  This
    diagnostic stores the latest bound so the switcher can apply hysteresis
    without owning the RHS or the integrator state.
    """

    def __init__(self) -> None:
        self.last: float = 0.0

    def update(self, jacobian: Tensor, h: float) -> float:
        """Update and return the Gershgorin bound for ``h * jacobian``."""
        n_rows = jacobian.shape[0]
        n_cols = jacobian.shape[1]
        max_row = 0.0
        for i in range(n_rows):
            row_sum = 0.0
            for j in range(n_cols):
                row_sum += abs(float(jacobian[i, j]))
            max_row = max(max_row, abs(h) * row_sum)
        self.last = max_row
        return max_row


class StiffnessSwitcher:
    """Hysteresis policy for Adams/BDF family selection.

    The policy switches Adams to BDF when the latest stiffness estimate exceeds
    ``stiff_threshold`` and switches BDF back to Adams only after it falls below
    ``nonstiff_threshold``.  The gap prevents method chatter near the boundary.
    """

    def __init__(
        self,
        *,
        stiff_threshold: float = 1.0,
        nonstiff_threshold: float = 0.5,
    ) -> None:
        if nonstiff_threshold >= stiff_threshold:
            raise ValueError("nonstiff_threshold must be below stiff_threshold.")
        self.stiff_threshold = stiff_threshold
        self.nonstiff_threshold = nonstiff_threshold

    def decide(self, current: FamilyName | str, stiffness: float) -> FamilySwitch:
        """Return the selected family for the latest stiffness estimate."""
        current = FamilyName(current)
        if current == "adams" and stiffness > self.stiff_threshold:
            return FamilySwitch(FamilyName("bdf"), stiffness, True)
        if current == "bdf" and stiffness < self.nonstiff_threshold:
            return FamilySwitch(FamilyName("adams"), stiffness, True)
        return FamilySwitch(current, stiffness, False)


__all__ = [
    "FamilyName",
    "FamilySwitch",
    "StiffnessDiagnostic",
    "StiffnessSwitcher",
]
