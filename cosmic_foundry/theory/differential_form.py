"""DifferentialForm: antisymmetric covariant tensor field of degree k."""

from __future__ import annotations

from abc import abstractmethod

from cosmic_foundry.theory.field import TensorField


class DifferentialForm(TensorField):  # noqa: B024
    """An antisymmetric (0, k)-tensor field on a smooth manifold M.

    A differential k-form assigns to each point p ∈ M a totally
    antisymmetric element of (T*M)^⊗k.  The degree k is the only free
    parameter; tensor_type is derived as (0, k).

    The de Rham complex Ω⁰ → Ω¹ → Ω² → … → Ωⁿ is the graded algebra
    of differential forms under the exterior derivative d.  Forms of
    degree 0 coincide with scalar fields; forms of degree 1 coincide
    with covector fields.  The first genuinely new cases are TwoForm
    (k=2) and above.

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


__all__ = ["DifferentialForm"]
