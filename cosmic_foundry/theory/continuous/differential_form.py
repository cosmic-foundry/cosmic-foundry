"""DifferentialForm: antisymmetric covariant tensor field of degree k.

The de Rham complex Ω⁰ → Ω¹ → … → Ωⁿ is graded by degree.
DifferentialForm is the ABC for this family.  Named subclasses ZeroForm,
OneForm, and TwoForm fix the degree and the Python value type for C,
enabling type-level discrimination of operator signatures.
"""

from __future__ import annotations

from abc import abstractmethod

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


class ZeroForm(DifferentialForm[D, sympy.Expr]):  # noqa: B024
    """A differential 0-form: a scalar field.

    Fixes degree = 0 and C = sympy.Expr, so the SymbolicFunction.__call__
    default (expr.subs) produces a scalar value.  This is the correct Python
    value type for a 0-form.

    Earns its class by deriving degree — the one degree of freedom present
    in DifferentialForm is removed.
    """

    @property
    def degree(self) -> int:
        return 0


class OneForm(DifferentialForm[D, tuple[sympy.Expr, ...]]):
    """A differential 1-form: a covector field.

    Fixes degree = 1 and C = tuple[sympy.Expr, ...].  The default
    SymbolicFunction.__call__ returns a single sympy.Expr (the expr
    evaluated at a point); concrete implementations may override __call__
    to return all n components.

    Earns its class by deriving degree and requiring component(axis).
    """

    @property
    def degree(self) -> int:
        return 1

    @abstractmethod
    def component(self, axis: int) -> sympy.Expr:
        """The axis-th coordinate component of this one-form."""


class TwoForm(DifferentialForm[D, sympy.Matrix]):  # noqa: B024
    """A differential 2-form: an antisymmetric rank-2 tensor field.

    Fixes degree = 2 and C = sympy.Matrix.  The antisymmetry condition
    T_{ij} = -T_{ji} must be enforced by concrete implementations.

    Earns its class by deriving degree.
    """

    @property
    def degree(self) -> int:
        return 2


__all__ = [
    "DifferentialForm",
    "OneForm",
    "TwoForm",
    "ZeroForm",
]
