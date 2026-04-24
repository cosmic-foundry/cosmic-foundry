"""DifferentialForm: antisymmetric covariant tensor field of degree k.

The de Rham complex Ω⁰ → Ω¹ → … → Ωⁿ is graded by degree.
DifferentialForm is the ABC for this family; concrete subclasses fix the
degree to name specific cases such as 0-forms (scalar fields) or 1-forms
(covector fields), but those named subclasses are implementation choices,
not distinct ABCs.
"""

from __future__ import annotations

from abc import abstractmethod

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


__all__ = [
    "DifferentialForm",
]
