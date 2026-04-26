"""CartesianExteriorDerivative: exact chain map on CartesianMesh."""

from __future__ import annotations

from typing import Any

import sympy

from cosmic_foundry.geometry.cartesian_mesh import CartesianMesh
from cosmic_foundry.theory.discrete.discrete_exterior_derivative import (
    DiscreteExteriorDerivative,
)
from cosmic_foundry.theory.discrete.discrete_field import DiscreteField
from cosmic_foundry.theory.discrete.edge_field import EdgeField, _CallableEdgeField
from cosmic_foundry.theory.discrete.face_field import FaceField, _CallableFaceField
from cosmic_foundry.theory.discrete.mesh import Mesh
from cosmic_foundry.theory.discrete.point_field import PointField
from cosmic_foundry.theory.discrete.volume_field import (
    VolumeField,
    _CallableVolumeField,
)


class CartesianExteriorDerivative(DiscreteExteriorDerivative):
    """Exact discrete exterior derivative on a CartesianMesh.

    Implements the three legs of the discrete de Rham chain complex:

        d₀ (degree=0): PointField → EdgeField
            (d₀φ)(a, v) = φ(v + eₐ) − φ(v)
            Indexed by (tangent_axis, low_vertex_idx).

        d₁ (degree=1): EdgeField → FaceField   [3-D only]
            (d₁A)(a, c) = A(b, v_base) + A(c̃, v_base+eᵦ)
                        − A(b, v_base+eꜯ) − A(c̃, v_base)
            where v_base = c with v_base[a] += 1, b=(a+1)%3, c̃=(a+2)%3.
            This is the standard Yee-grid discrete curl.

        d₂ (degree=2): FaceField → VolumeField
            (d₂F)(c) = Σₐ [F(a, c) − F(a, c − eₐ)]
            Raw combinatorial divergence (no volume normalization).

    d_{k+1} ∘ d_k = 0 exactly for all k.

    Parameters
    ----------
    mesh:
        The CartesianMesh on which the complex is defined.
    degree:
        Input form degree k ∈ {0, 1, 2}.  degree=1 requires ndim=3.
    """

    def __init__(self, mesh: CartesianMesh, degree: int) -> None:
        ndim = len(mesh._shape)
        if degree not in (0, 1, 2):
            raise ValueError(
                f"CartesianExteriorDerivative degree must be 0, 1, or 2; got {degree}"
            )
        if degree == 1 and ndim != 3:
            raise ValueError(
                f"CartesianExteriorDerivative d₁ requires ndim=3; got {ndim}"
            )
        self._mesh = mesh
        self._degree = degree

    @property
    def mesh(self) -> Mesh:
        return self._mesh

    @property
    def degree(self) -> int:
        return self._degree

    def __call__(self, field: DiscreteField[Any]) -> DiscreteField[Any]:
        if self._degree == 0:
            return self._d0(field)  # type: ignore[arg-type]
        if self._degree == 1:
            return self._d1(field)  # type: ignore[arg-type]
        return self._d2(field)  # type: ignore[arg-type]

    def _d0(self, phi: PointField[sympy.Expr]) -> EdgeField[sympy.Expr]:
        """Gradient: (d₀φ)(a, v) = φ(v + eₐ) − φ(v)."""
        mesh = self._mesh

        def compute(edge: tuple[int, tuple[int, ...]]) -> sympy.Expr:
            axis, v_idx = edge
            v_hi = v_idx[:axis] + (v_idx[axis] + 1,) + v_idx[axis + 1 :]
            return phi(v_hi) - phi(v_idx)

        return _CallableEdgeField(mesh, compute)

    def _d1(self, A: EdgeField[sympy.Expr]) -> FaceField[sympy.Expr]:
        """Curl (3-D): boundary circulation of A around each face."""
        mesh = self._mesh

        def compute(face: tuple[int, tuple[int, ...]]) -> sympy.Expr:
            a, c_idx = face
            # v_base: vertex at the low corner of the face (cell index → vertex index)
            v_base = tuple(c_idx[i] + (1 if i == a else 0) for i in range(3))
            b = (a + 1) % 3
            c = (a + 2) % 3
            v_base_b = v_base[:b] + (v_base[b] + 1,) + v_base[b + 1 :]
            v_base_c = v_base[:c] + (v_base[c] + 1,) + v_base[c + 1 :]
            return A((b, v_base)) + A((c, v_base_b)) - A((b, v_base_c)) - A((c, v_base))

        return _CallableFaceField(mesh, compute)

    def _d2(self, F: FaceField[sympy.Expr]) -> VolumeField[sympy.Expr]:
        """Divergence: (d₂F)(c) = Σₐ [F(a, c) − F(a, c − eₐ)]."""
        mesh = self._mesh
        ndim = len(mesh._shape)

        def compute(c_idx: tuple[int, ...]) -> sympy.Expr:
            return sum(
                F((a, c_idx)) - F((a, c_idx[:a] + (c_idx[a] - 1,) + c_idx[a + 1 :]))
                for a in range(ndim)
            )

        return _CallableVolumeField(mesh, compute)


__all__ = ["CartesianExteriorDerivative"]
