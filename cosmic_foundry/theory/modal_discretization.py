"""ModalDiscretization ABC."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from cosmic_foundry.theory.discretization import Discretization


class ModalDiscretization(Discretization):
    """A Discretization whose DOFs are coefficients of basis functions on M.

    Each index i ∈ I labels a basis function bᵢ: M → ℝ.  A field is
    represented as f ≈ Σᵢ cᵢ bᵢ, where cᵢ are the modal coefficients
    stored as DOFs.  The DOFs do not have positions in M — they have
    associated basis functions.

    Spectral methods (Fourier, Chebyshev, spherical harmonics, etc.) are
    ModalDiscretizations.  The index set I is typically the set of
    retained wavenumbers or mode indices up to some truncation.

    Required:
        basis_functions — description of the basis; concrete type depends
                          on the specific spectral method
    """

    @property
    @abstractmethod
    def basis_functions(self) -> Any:
        """Description of the basis functions indexed by I."""


__all__ = [
    "ModalDiscretization",
]
