"""Operator-splitting integrators: Lie and Strang."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from cosmic_foundry.computation.time_integrators.integrator import (
    RHSProtocol,
    RKState,
    TimeIntegrator,
)


@runtime_checkable
class OperatorSplitRHSProtocol(Protocol):
    """Protocol for operator-split ODEs ``du/dt = f_1(t,u) + … + f_k(t,u)``.

    The ODE right-hand side is partitioned into ``k`` components, each a
    ``RHSProtocol``.  A splitting integrator advances each component in turn
    using a dedicated sub-integrator; the full step is the composition of those
    sub-steps according to a ``SplittingStep`` sequence.
    """

    @property
    def components(self) -> Sequence[RHSProtocol]:
        """Ordered list of sub-RHS objects ``[f_1, …, f_k]``."""
        ...


class OperatorSplitRHS:
    """Concrete ``OperatorSplitRHSProtocol`` from a list of sub-RHS callables."""

    def __init__(self, components: Sequence[RHSProtocol]) -> None:
        self._components: list[RHSProtocol] = list(components)

    @property
    def components(self) -> Sequence[RHSProtocol]:
        return self._components


@dataclasses.dataclass(frozen=True)
class SplittingStep:
    """One substep in an operator-splitting sequence.

    ``component_index`` selects which component RHS (and matching
    sub-integrator) to use; ``weight`` is the fraction of the full ``dt`` for
    this substep.  Negative weights are permitted and occur in Yoshida
    composition.
    """

    component_index: int
    weight: float


class StrangSplittingIntegrator:
    """Meta-integrator that composes sub-integrators via a splitting sequence.

    Each ``SplittingStep`` in ``sequence`` advances
    ``rhs.components[s.component_index]`` by ``s.weight * dt`` using
    ``sub_integrators[s.component_index]``.  Sub-integrators must accept and
    return ``RKState``; passing the same integrator at multiple positions is
    allowed.

    ``order`` is declared by the factory function that constructs the sequence
    (``lie_steps``, ``strang_steps``); it is not derived from the sequence
    itself.

    Parameters
    ----------
    sub_integrators:
        One integrator per component.  ``sub_integrators[i]`` is used for all
        substeps whose ``component_index`` is ``i``.
    sequence:
        Ordered list of ``SplittingStep`` objects defining the splitting method.
    order:
        Declared convergence order of the splitting scheme.
    """

    def __init__(
        self,
        sub_integrators: Sequence[TimeIntegrator],
        sequence: Sequence[SplittingStep],
        order: int,
    ) -> None:
        self._sub_integrators = list(sub_integrators)
        self._sequence = list(sequence)
        self._order = order

    @property
    def order(self) -> int:
        """Declared convergence order of the splitting scheme."""
        return self._order

    def step(
        self,
        rhs: OperatorSplitRHSProtocol,
        state: RKState,
        dt: float,
    ) -> RKState:
        """Advance ``state`` by one full step of size ``dt``.

        Applies each ``SplittingStep`` in ``self._sequence`` in order.  The
        output ``RKState`` of each substep feeds into the next.  The ``dt``
        and ``err`` fields of the returned state reflect the last substep only;
        the ``t`` field is ``state.t + dt``.
        """
        for substep in self._sequence:
            sub_rhs = rhs.components[substep.component_index]
            sub_integrator = self._sub_integrators[substep.component_index]
            state = sub_integrator.step(sub_rhs, state, substep.weight * dt)
        return state


def lie_steps() -> list[SplittingStep]:
    """Lie (sequential) splitting sequence for two components.

    ``[A(h), B(h)]``: advance component 0 by the full step, then component 1.
    First-order accurate; error is ``O(h²)`` per step, ``O(h)`` globally.
    """
    return [SplittingStep(0, 1.0), SplittingStep(1, 1.0)]


def strang_steps() -> list[SplittingStep]:
    """Strang (symmetric) splitting sequence for two components.

    ``[A(h/2), B(h), A(h/2)]``: half-step of component 0, full step of
    component 1, half-step of component 0.  Second-order accurate; the
    palindrome structure cancels the leading commutator error term.
    """
    return [SplittingStep(0, 0.5), SplittingStep(1, 1.0), SplittingStep(0, 0.5)]


__all__ = [
    "OperatorSplitRHS",
    "OperatorSplitRHSProtocol",
    "SplittingStep",
    "StrangSplittingIntegrator",
    "lie_steps",
    "strang_steps",
]
