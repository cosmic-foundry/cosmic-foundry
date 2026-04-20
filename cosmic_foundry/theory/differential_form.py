"""DifferentialForm: antisymmetric covariant tensor field of degree k.

The de Rham complex Ω⁰ → Ω¹ → … → Ωⁿ is graded by degree:

  DifferentialForm  — abstract; degree k is the only free parameter
  ScalarField       — Ω⁰(M) = C∞(M); degree 0, tensor type (0, 0)
  CovectorField     — Ω¹(M) = Γ(T*M); degree 1, tensor type (0, 1)

ScalarField and CovectorField are placed here (not in field.py) because
they are specific cases of the de Rham complex, not independent concepts:
a scalar field *is* a 0-form and a covector field *is* a 1-form.
The antisymmetry condition is vacuous at degrees 0 and 1 (no index pairs
to antisymmetrize), but the type relationship is real and matters for any
operation that expects a differential form (e.g., integration, exterior
derivative).
"""

from __future__ import annotations

from abc import abstractmethod

from cosmic_foundry.theory.field import TensorField


class DifferentialForm(TensorField):  # noqa: B024
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


class ScalarField(DifferentialForm):  # noqa: B024
    """A scalar field: Ω⁰(M) = C∞(M), degree 0, tensor type (0, 0).

    A 0-form assigns a real number to every point; it is the base case of
    the de Rham complex.  The antisymmetry condition is vacuous (no indices).
    """

    @property
    def degree(self) -> int:
        return 0


class CovectorField(DifferentialForm):  # noqa: B024
    """A covector field (1-form): Ω¹(M) = Γ(T*M), degree 1, tensor type (0, 1).

    A 1-form assigns a linear functional on T_pM to every point p.  It is
    the degree-1 case of the de Rham complex; the antisymmetry condition is
    again vacuous (a single-index tensor has nothing to antisymmetrize).
    """

    @property
    def degree(self) -> int:
        return 1


__all__ = [
    "CovectorField",
    "DifferentialForm",
    "ScalarField",
]
