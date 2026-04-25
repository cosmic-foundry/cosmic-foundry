"""Instance registry for convergence tests.

Each block here does two things:
  1. imports the concrete class (making it visible to __subclasses__())
  2. registers its test instances in CONVERGENCE_INSTANCES
"""

from __future__ import annotations

from typing import Any

import sympy

from cosmic_foundry.geometry.cartesian_mesh import CartesianMesh
from cosmic_foundry.geometry.diffusive_flux import DiffusiveFlux
from cosmic_foundry.geometry.euclidean_manifold import EuclideanManifold
from cosmic_foundry.geometry.fvm_discretization import (
    FVMDiscretization,
    _AssembledFVMOperator,
)
from cosmic_foundry.theory.continuous.differential_form import ZeroForm
from cosmic_foundry.theory.continuous.diffusion_operator import DiffusionOperator
from cosmic_foundry.theory.continuous.manifold import Manifold
from cosmic_foundry.theory.continuous.poisson_equation import PoissonEquation
from tests.support.convergence_registry import CONVERGENCE_INSTANCES


class _ZeroFormField(ZeroForm[Any]):
    def __init__(
        self,
        manifold: Any,
        expr: sympy.Expr,
        symbols: tuple[sympy.Symbol, ...],
    ) -> None:
        self._manifold = manifold
        self._expr = expr
        self._symbols = symbols

    @property
    def manifold(self) -> Any:
        return self._manifold

    @property
    def expr(self) -> sympy.Expr:
        return self._expr

    @property
    def symbols(self) -> tuple[sympy.Symbol, ...]:
        return self._symbols


class _ConcretePoissonEquation(PoissonEquation):
    def __init__(self, manifold: Manifold, source: ZeroForm) -> None:
        self._manifold = manifold
        self._source = source

    @property
    def manifold(self) -> Manifold:
        return self._manifold

    @property
    def source(self) -> ZeroForm:
        return self._source


_manifold = EuclideanManifold(1)
_x = _manifold.atlas[0].symbols[0]
_diffusion_op = DiffusionOperator(_manifold)
_lo = DiffusiveFlux.min_order
_step = DiffusiveFlux.order_step

# --- DiffusiveFlux instances ---
CONVERGENCE_INSTANCES[DiffusiveFlux] = [
    DiffusiveFlux(_lo, _diffusion_op),
    DiffusiveFlux(_lo + _step, _diffusion_op),
]

# --- _AssembledFVMOperator instances ---
_zero_source = _ZeroFormField(_manifold, sympy.Integer(0), (_x,))
_poisson = _ConcretePoissonEquation(_manifold, _zero_source)
_dummy_mesh = CartesianMesh(
    origin=(sympy.Integer(0),),
    spacing=(sympy.Integer(1),),
    shape=(4,),
)
CONVERGENCE_INSTANCES[_AssembledFVMOperator] = [
    FVMDiscretization(_dummy_mesh, DiffusiveFlux(_lo, _diffusion_op))(_poisson),
    FVMDiscretization(_dummy_mesh, DiffusiveFlux(_lo + _step, _diffusion_op))(_poisson),
]
