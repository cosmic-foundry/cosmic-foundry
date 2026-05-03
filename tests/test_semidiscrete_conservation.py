"""Semi-discrete conservation calculation claims."""

from __future__ import annotations

import math
from typing import Any

import pytest

import cosmic_foundry.computation.time_integrators as _ti
from cosmic_foundry.computation.algorithm_capabilities import (
    AlgorithmRequest,
    MapStructureField,
    ParameterDescriptor,
    map_structure_parameter_schema,
)
from cosmic_foundry.computation.backends import NumpyBackend
from cosmic_foundry.computation.tensor import Tensor
from cosmic_foundry.computation.time_integrators.capabilities import (
    conserved_rhs_evaluation_descriptor,
    time_integration_step_map_regions,
)
from tests.claims import Claim
from tests.selection_ownership import SelectionOwnership

_TIME_BACKEND = NumpyBackend()


def _step_map_ownership(descriptor: ParameterDescriptor) -> SelectionOwnership:
    return SelectionOwnership(
        descriptor,
        time_integration_step_map_regions(),
        map_structure_parameter_schema(),
    )


def _sum_entries(u: Tensor) -> float:
    return sum(float(u[i]) for i in range(u.shape[0]))


class _PeriodicAdvectionConservationClaim(Claim[Any]):
    """Grounded claim for conservation evidence on a semi-discrete map."""

    @property
    def description(self) -> str:
        return "correctness/periodic_advection_conservation"

    def check(self, _calibration: Any) -> None:
        cell_count = 64
        speed = 1.0
        dx = 1.0 / cell_count
        dt = 0.25 * dx / speed
        steps = 64
        phase = speed * dt * steps
        descriptor = conserved_rhs_evaluation_descriptor(1)

        def profile(x: float) -> float:
            return 1.0 + 0.2 * math.sin(2.0 * math.pi * x)

        def periodic(x: float) -> float:
            return x - math.floor(x)

        initial_values = [profile((i + 0.5) * dx) for i in range(cell_count)]

        def rhs(_t: float, u: Tensor) -> Tensor:
            return Tensor(
                [
                    -speed * (float(u[i]) - float(u[(i - 1) % cell_count])) / dx
                    for i in range(cell_count)
                ],
                backend=u.backend,
            )

        assert (
            descriptor.coordinate(MapStructureField.CONSERVED_LINEAR_FORM_COUNT).value
            == 1
        )
        assert (
            _ti.select_time_integrator(
                AlgorithmRequest(
                    requested_properties=frozenset({"one_step"}),
                    order=4,
                    descriptor=descriptor,
                )
            ).implementation
            == _step_map_ownership(descriptor).owner.__name__
        )
        _step_map_ownership(descriptor).assert_owned_cell()

        state = _ti.ODEState(0.0, Tensor(initial_values, backend=_TIME_BACKEND))
        initial_total = _sum_entries(state.u)
        assert abs(_sum_entries(rhs(0.0, state.u))) < 1.0e-12
        integrator = _ti.RungeKuttaIntegrator(4)
        for _ in range(steps):
            state = integrator.step(_ti.BlackBoxRHS(rhs), state, dt)

        exact = [profile(periodic((i + 0.5) * dx - phase)) for i in range(cell_count)]
        max_error = max(abs(float(state.u[i]) - exact[i]) for i in range(cell_count))

        assert abs(_sum_entries(state.u) - initial_total) < 1.0e-12
        assert max_error < 2.5e-2


_CORRECT_CLAIMS: tuple[Claim[Any], ...] = (_PeriodicAdvectionConservationClaim(),)


@pytest.mark.parametrize(
    "claim", _CORRECT_CLAIMS, ids=[c.description for c in _CORRECT_CLAIMS]
)
def test_correctness(claim: Claim[Any]) -> None:
    claim.check(None)
