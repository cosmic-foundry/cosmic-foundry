"""Shared Newton-iteration kernel for implicit integrators.

Both ``DIRKIntegrator`` and ``IMEXIntegrator`` solve the same fixed-point
problem at each stage::

    y − γ·h·f(y) = y_exp

via Newton iteration with LU factorization.  The only caller-visible
difference is which callable plays the role of ``f`` (the full RHS for
DIRK, the implicit component for IMEX).  ``newton_solve`` accepts both
as plain callables so the kernel stays independent of the RHS protocol.
"""

from __future__ import annotations

from collections.abc import Callable

from cosmic_foundry.computation.decompositions.lu_factorization import LUFactorization
from cosmic_foundry.computation.tensor import Tensor, norm

_LU = LUFactorization()
_NEWTON_MAX_ITER = 50
_NEWTON_TOL = 1e-12


def newton_solve(
    y_exp: Tensor,
    gamma_dt: float,
    f: Callable[[Tensor], Tensor],
    jac: Callable[[Tensor], Tensor],
) -> Tensor:
    """Solve ``y − gamma_dt·f(y) = y_exp`` by Newton iteration.

    Terminates when either the residual norm or the step norm falls below
    ``_NEWTON_TOL * (1 + ‖y‖)``, or after ``_NEWTON_MAX_ITER`` iterations.

    Parameters
    ----------
    y_exp:
        Explicit right-hand side of the stage equation.
    gamma_dt:
        Product of the diagonal coefficient and the step size (γᵢ · h).
    f:
        Callable ``y ↦ f(y)``; the nonlinear term to be treated implicitly.
    jac:
        Callable ``y ↦ ∂f/∂y``; the Jacobian of ``f``.

    Returns
    -------
    Tensor
        Converged stage value ``y``.
    """
    backend = y_exp.backend
    n = y_exp.shape[0]
    y = y_exp
    for _ in range(_NEWTON_MAX_ITER):
        fy = f(y)
        r = y - gamma_dt * fy - y_exp
        if float(norm(r)) < _NEWTON_TOL * (1.0 + float(norm(y))):
            break
        J = jac(y)
        M = Tensor.eye(n, backend=backend) - gamma_dt * J
        delta = _LU.factorize(M).solve(Tensor.zeros(n, backend=backend) - r)
        y = y + delta
        if float(norm(delta)) < _NEWTON_TOL * (1.0 + float(norm(y))):
            break
    return y


__all__ = ["newton_solve"]
