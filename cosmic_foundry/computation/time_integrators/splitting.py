"""Operator-splitting integrators: Lie, Strang, and Yoshida compositions."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from cosmic_foundry.computation.time_integrators.integrator import (
    ODEState,
    RHSProtocol,
    TimeIntegrator,
)


@runtime_checkable
class CompositeRHSProtocol(Protocol):
    """Protocol for operator-split ODEs ``du/dt = f_1(t,u) + … + f_k(t,u)``.

    The ODE right-hand side is partitioned into ``k`` components, each a
    ``RHSProtocol``.  A splitting integrator advances each component in turn
    using a dedicated sub-integrator; the full step is the composition of those
    sub-steps according to a splitting sequence.
    """

    @property
    def components(self) -> Sequence[RHSProtocol]:
        """Ordered list of sub-RHS objects ``[f_1, …, f_k]``."""
        ...


class CompositeRHS:
    """Concrete ``CompositeRHSProtocol`` from a list of sub-RHS callables."""

    def __init__(self, components: Sequence[RHSProtocol]) -> None:
        self._components: list[RHSProtocol] = list(components)

    @property
    def components(self) -> Sequence[RHSProtocol]:
        return self._components


@dataclasses.dataclass(frozen=True)
class _SplittingStep:
    """One substep in an operator-splitting sequence (internal)."""

    component_index: int
    weight: float


def _strang_steps() -> list[_SplittingStep]:
    """Strang (symmetric) splitting sequence for two components.

    ``[A(h/2), B(h), A(h/2)]``: half-step of component 0, full step of
    component 1, half-step of component 0.  Second-order; palindrome structure
    cancels the leading commutator error term.
    """
    return [_SplittingStep(0, 0.5), _SplittingStep(1, 1.0), _SplittingStep(0, 0.5)]


def _merge_steps(steps: list[_SplittingStep]) -> list[_SplittingStep]:
    """Merge adjacent substeps for the same component."""
    merged: list[_SplittingStep] = []
    for step in steps:
        if merged and merged[-1].component_index == step.component_index:
            previous = merged[-1]
            merged[-1] = _SplittingStep(
                previous.component_index, previous.weight + step.weight
            )
        else:
            merged.append(step)
    return [step for step in merged if step.weight != 0.0]


def _scale_steps(steps: list[_SplittingStep], scale: float) -> list[_SplittingStep]:
    """Return ``steps`` with every time-weight multiplied by ``scale``."""
    return [_SplittingStep(step.component_index, scale * step.weight) for step in steps]


def _triple_jump_steps(
    base: list[_SplittingStep],
    *,
    exponent: float,
) -> list[_SplittingStep]:
    """Lift a symmetric splitting by two orders via Yoshida triple jump."""
    w1 = 1.0 / (2.0 - 2.0**exponent)
    w0 = 1.0 - 2.0 * w1
    return _merge_steps(
        _scale_steps(base, w1) + _scale_steps(base, w0) + _scale_steps(base, w1)
    )


def _yoshida4_steps() -> list[_SplittingStep]:
    """Yoshida 4th-order triple-jump splitting sequence for two components.

    Constructs S₄(h) = S₂(w₁h) ∘ S₂(w₀h) ∘ S₂(w₁h) where S₂ is the
    Strang splitting and the weights are

        w₁ = 1 / (2 − 2^{1/3}),   w₀ = 1 − 2 w₁.

    The two internal A half-steps at the junction of adjacent Strang blocks
    collapse into single steps, giving the 7-step sequence::

        A(w₁/2)  B(w₁)  A((w₁+w₀)/2)  B(w₀)  A((w₀+w₁)/2)  B(w₁)  A(w₁/2)

    w₀ ≈ −1.702, so steps 3, 4, and 5 have negative weights.  Sub-integrators
    must handle negative ``dt`` correctly; this holds for any method applied to
    a time-reversible operator.
    """
    return _triple_jump_steps(_strang_steps(), exponent=1.0 / 3.0)


def _yoshida6_steps() -> list[_SplittingStep]:
    """Yoshida 6th-order triple-jump composition of the 4th-order splitter."""
    return _triple_jump_steps(_yoshida4_steps(), exponent=1.0 / 5.0)


_SPLITTING_SEQUENCES: dict[int, list[_SplittingStep]] = {
    1: [_SplittingStep(0, 1.0), _SplittingStep(1, 1.0)],
    2: _strang_steps(),
    4: _yoshida4_steps(),
    6: _yoshida6_steps(),
}


class CompositionIntegrator(TimeIntegrator):
    """Meta-integrator that composes sub-integrators via a splitting sequence.

    Each substep in the sequence advances one component of the composite RHS
    by a weighted fraction of the full ``dt`` using the matching sub-integrator.
    Sub-integrators must accept and return ``ODEState``; passing the same
    integrator at multiple positions is allowed.

    The splitting sequence is selected by ``order``:
        1 — Lie splitting (AB)
        2 — Strang symmetric splitting (ABA)
        4 — Yoshida triple-jump composition of order-2 Strang
        6 — Yoshida triple-jump composition of order-4 splitting

    Parameters
    ----------
    sub_integrators:
        One integrator per component.  ``sub_integrators[i]`` is used for all
        substeps whose component index is ``i``.
    order:
        Declared convergence order of the splitting scheme.  Must be one of
        {1, 2, 4, 6}.
    """

    def __init__(
        self,
        sub_integrators: Sequence[TimeIntegrator],
        order: int,
    ) -> None:
        if order not in _SPLITTING_SEQUENCES:
            raise ValueError(
                f"CompositionIntegrator order must be one of "
                f"{sorted(_SPLITTING_SEQUENCES)}, got {order}"
            )
        self._sub_integrators = list(sub_integrators)
        self._sequence = _SPLITTING_SEQUENCES[order]
        self._order = order

    @property
    def order(self) -> int:
        """Declared convergence order of the splitting scheme."""
        return self._order

    def step(
        self,
        rhs: CompositeRHSProtocol,
        state: ODEState,
        dt: float,
    ) -> ODEState:
        """Advance ``state`` by one full step of size ``dt``.

        Applies each substep in the sequence in order.  The output
        ``ODEState`` of each substep feeds into the next.  The ``t`` field of
        the returned state is ``state.t + dt``.
        """
        t_start = state.t
        for substep in self._sequence:
            sub_rhs = rhs.components[substep.component_index]
            sub_integrator = self._sub_integrators[substep.component_index]
            state = sub_integrator.step(sub_rhs, state, substep.weight * dt)
        return ODEState(t_start + dt, state.u, state.dt, state.err)


__all__ = [
    "CompositeRHS",
    "CompositeRHSProtocol",
    "CompositionIntegrator",
]
