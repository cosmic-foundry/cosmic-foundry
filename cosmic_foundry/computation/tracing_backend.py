"""TracingBackend: AbstractBackend used for JIT-compatibility CI."""

from __future__ import annotations

from cosmic_foundry.computation.abstract_backend import AbstractBackend


class TracingBackend(AbstractBackend):
    """Backend for verifying that algorithm code contains no accidental
    materialization.

    Construct abstract Tensors with this backend, run the function under
    test, and assert that no JitIncompatibleError is raised.  Any call to
    float(tensor), bool(tensor), or tensor.item() inside the traced function
    surfaces immediately with a traceback pointing to the violation.

    Example
    -------
    from cosmic_foundry.computation.abstract_value import AbstractValue
    from cosmic_foundry.computation.tensor import Tensor

    b = TracingBackend()
    a = Tensor._wrap(AbstractValue((4, 4)), b)
    rhs = Tensor._wrap(AbstractValue((4,)), b)
    solver = DenseJacobiSolver()
    state = solver.init_state(a, rhs)
    solver.step(state)       # raises JitIncompatibleError if not JIT-compatible
    solver.converged(state)  # same
    """


__all__ = ["TracingBackend"]
