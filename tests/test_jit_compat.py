"""JIT-compatibility CI: verify that compilable functions contain no accidental
materialization.

Each test constructs abstract Tensors backed by TracingBackend and runs a
function that must be JIT-compatible.  Any call to float(tensor), bool(tensor),
or tensor.item() inside the traced function raises JitIncompatibleError,
pinpointing the exact violation.

Functions tested here are those intended to run inside a compiled loop (step,
converged) or to be compiled as a whole (factorize, factored_solve).  The outer
solve() loop is intentionally excluded — it holds the materialization boundary
and is expected to call .item().
"""

from __future__ import annotations

import pytest

from cosmic_foundry.computation.abstract_value import (
    AbstractValue,
    JitIncompatibleError,
)
from cosmic_foundry.computation.dense_jacobi_solver import DenseJacobiSolver
from cosmic_foundry.computation.lu_factorization import LUFactorization
from cosmic_foundry.computation.tensor import Tensor
from cosmic_foundry.computation.tracing_backend import TracingBackend

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _abstract(shape: tuple[int, ...]) -> Tensor:
    """Abstract Tensor of given shape backed by a fresh TracingBackend."""
    return Tensor._wrap(AbstractValue(shape), TracingBackend())


def _abstract_on(shape: tuple[int, ...], b: TracingBackend) -> Tensor:
    return Tensor._wrap(AbstractValue(shape), b)


# ---------------------------------------------------------------------------
# DenseJacobiSolver
# ---------------------------------------------------------------------------


class TestJacobiJitCompatible:
    def _state(self, n: int = 4) -> tuple[DenseJacobiSolver, object]:
        b = TracingBackend()
        a = _abstract_on((n, n), b)
        rhs = _abstract_on((n,), b)
        solver = DenseJacobiSolver()
        return solver, solver.init_state(a, rhs)

    def test_init_state(self) -> None:
        """init_state itself must not materialise: omega stays a 0-d Tensor."""
        b = TracingBackend()
        a = _abstract_on((4, 4), b)
        rhs = _abstract_on((4,), b)
        DenseJacobiSolver().init_state(a, rhs)

    def test_step(self) -> None:
        solver, state = self._state()
        solver.step(state)

    def test_converged(self) -> None:
        solver, state = self._state()
        solver.converged(state)

    def test_converged_returns_tensor(self) -> None:
        solver, state = self._state()
        result = solver.converged(state)
        assert isinstance(result, Tensor)
        assert result.shape == ()

    def test_step_preserves_state_structure(self) -> None:
        solver, state = self._state()
        new_state = solver.step(state)
        assert type(new_state) is type(state)


# ---------------------------------------------------------------------------
# LUFactorization / LUFactoredMatrix
# ---------------------------------------------------------------------------


class TestLUJitCompatible:
    def test_factorize(self) -> None:
        b = TracingBackend()
        a = _abstract_on((4, 4), b)
        LUFactorization().factorize(a)

    def test_factored_solve(self) -> None:
        b = TracingBackend()
        a = _abstract_on((4, 4), b)
        rhs = _abstract_on((4,), b)
        factored = LUFactorization().factorize(a)
        factored.solve(rhs)

    def test_factorize_3x3(self) -> None:
        b = TracingBackend()
        a = _abstract_on((3, 3), b)
        LUFactorization().factorize(a)

    def test_factorize_1x1(self) -> None:
        b = TracingBackend()
        a = _abstract_on((1, 1), b)
        LUFactorization().factorize(a)


# ---------------------------------------------------------------------------
# Regression: materialisation is caught
# ---------------------------------------------------------------------------


class TestMaterialisationIsDetected:
    """Confirm that JitIncompatibleError fires for known bad patterns."""

    def test_float_of_tensor_raises(self) -> None:
        t = _abstract((4,))
        with pytest.raises(JitIncompatibleError):
            float(t[0])

    def test_bool_of_tensor_raises(self) -> None:
        t = _abstract(())
        with pytest.raises(JitIncompatibleError):
            bool(t)

    def test_item_raises(self) -> None:
        t = _abstract(())
        with pytest.raises(JitIncompatibleError):
            t.item()

    def test_norm_raises(self) -> None:
        t = _abstract((4,))
        with pytest.raises(JitIncompatibleError):
            t.norm()
