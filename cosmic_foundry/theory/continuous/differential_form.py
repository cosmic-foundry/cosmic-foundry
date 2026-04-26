"""DifferentialForm: antisymmetric covariant tensor field of degree k.

The de Rham complex Ω⁰ → Ω¹ → … → Ωⁿ is graded by degree.
DifferentialForm is the ABC for this family.  Named subclasses ZeroForm,
OneForm, and TwoForm fix the degree and the Python value type for C,
enabling type-level discrimination of operator signatures.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import sympy

from cosmic_foundry.theory.continuous.field import C, D, TensorField


class DifferentialForm(TensorField[D, C]):  # noqa: B024
    """An antisymmetric (0, k)-tensor field on a smooth manifold M.

    A differential k-form assigns to each point p ∈ M a totally
    antisymmetric element of (T*M)^⊗k.  The degree k is the only free
    parameter; tensor_type is derived as (0, k).

    Required:
        degree    — the degree k ∈ {0, 1, …, ndim(M)}
        manifold  — the smooth manifold on which this form is defined
                    (inherited abstract from TensorField)
    """

    @property
    @abstractmethod
    def degree(self) -> int:
        """The degree k of this differential k-form."""

    @property
    def tensor_type(self) -> tuple[int, int]:
        return (0, self.degree)


class ZeroForm(DifferentialForm[D, sympy.Expr]):
    """A differential 0-form: a scalar field.

    Fixes degree = 0 and C = sympy.Expr.  Concrete: ZeroForm(manifold,
    expr, symbols) constructs a scalar field directly.  Subclass when the
    field is computed or derived rather than given as a stored expression.
    """

    def __init__(
        self,
        manifold: D,
        expr: sympy.Expr,
        symbols: tuple[sympy.Symbol, ...],
    ) -> None:
        self._manifold = manifold
        self._expr = expr
        self._symbols = symbols

    @property
    def degree(self) -> int:
        return 0

    @property
    def manifold(self) -> D:
        return self._manifold

    @property
    def expr(self) -> sympy.Expr:
        return self._expr

    @property
    def symbols(self) -> tuple[sympy.Symbol, ...]:
        return self._symbols


class OneForm(DifferentialForm[D, tuple[sympy.Expr, ...]]):
    """A differential 1-form: a covector field.

    Fixes degree = 1 and C = tuple[sympy.Expr, ...].  Concrete:
    OneForm(manifold, components, symbols) constructs a covector field
    directly.  Subclass when the components are computed or derived.
    component(axis) returns components[axis]; expr returns components[0].
    """

    def __init__(
        self,
        manifold: D,
        components: tuple[sympy.Expr, ...],
        symbols: tuple[sympy.Symbol, ...],
    ) -> None:
        self._manifold = manifold
        self._components = components
        self._symbols = symbols

    @property
    def degree(self) -> int:
        return 1

    @property
    def manifold(self) -> D:
        return self._manifold

    @property
    def expr(self) -> sympy.Expr:
        return self._components[0]

    @property
    def symbols(self) -> tuple[sympy.Symbol, ...]:
        return self._symbols

    def component(self, axis: int) -> sympy.Expr:
        """The axis-th coordinate component of this one-form."""
        return self._components[axis]


class TwoForm(DifferentialForm[D, sympy.Matrix]):
    """A differential 2-form: an antisymmetric rank-2 tensor field.

    Fixes degree = 2 and C = sympy.Matrix.  Concrete: TwoForm(manifold,
    matrix, symbols) constructs a 2-form directly.  The antisymmetry
    condition T_{ij} = -T_{ji} must be enforced by the caller.
    expr returns matrix[0, 0] as the primary scalar component.
    """

    def __init__(
        self,
        manifold: D,
        matrix: sympy.Matrix,
        symbols: tuple[sympy.Symbol, ...],
    ) -> None:
        self._manifold: Any = manifold
        self._matrix = matrix
        self._symbols = symbols

    @property
    def degree(self) -> int:
        return 2

    @property
    def manifold(self) -> Any:
        return self._manifold

    @property
    def expr(self) -> sympy.Expr:
        return self._matrix[0, 0]

    @property
    def symbols(self) -> tuple[sympy.Symbol, ...]:
        return self._symbols

    def component(self, i: int, j: int) -> sympy.Expr:
        """The (i, j) component of this two-form (antisymmetric matrix entry)."""
        return self._matrix[i, j]


class ThreeForm(DifferentialForm[D, sympy.Expr]):
    """A differential 3-form: a volume form (scalar density).

    Fixes degree = 3 and C = sympy.Expr.  In three spatial dimensions a
    3-form has a single independent component f, representing the density
    f dV.  Concrete: ThreeForm(manifold, expr, symbols) constructs a
    volume-form field directly.  Subclass when the density is computed or
    derived rather than stored as an expression.

    ThreeForm is the continuous counterpart of VolumeField in the discrete
    layer: both represent Ωⁿ (volume-form) quantities, with VolumeField
    storing the cell total ∫_Ω f dV.
    """

    def __init__(
        self,
        manifold: D,
        expr: sympy.Expr,
        symbols: tuple[sympy.Symbol, ...],
    ) -> None:
        self._manifold = manifold
        self._expr = expr
        self._symbols = symbols

    @property
    def degree(self) -> int:
        return 3

    @property
    def manifold(self) -> D:
        return self._manifold

    @property
    def expr(self) -> sympy.Expr:
        return self._expr

    @property
    def symbols(self) -> tuple[sympy.Symbol, ...]:
        return self._symbols


__all__ = [
    "DifferentialForm",
    "OneForm",
    "ThreeForm",
    "TwoForm",
    "ZeroForm",
]
