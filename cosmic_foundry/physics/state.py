"""State: concrete Tensor-backed VolumeField[float]."""

from __future__ import annotations

from cosmic_foundry.computation.tensor import Tensor
from cosmic_foundry.theory.discrete.mesh import Mesh
from cosmic_foundry.theory.discrete.volume_field import VolumeField


class State(VolumeField[float]):
    """Concrete Tensor-backed discrete scalar field.

    State is the simulation-state object: a VolumeField[float] whose values
    are total integrals over each cell, stored in a flat Tensor of shape
    (n_cells,).  Index mapping follows axis-0-fastest ordering:
    flat = Σ_a idx[a] · stride[a], where stride[0] = 1 and
    stride[a] = Π_{k<a} shape[k].

    When backed by PythonBackend, the Tensor leaves may hold any Python
    object (e.g. sympy.Expr for symbolic computation); __call__ returns
    the raw leaf value without casting.

    Parameters
    ----------
    mesh:
        The mesh on which the field is defined; determines shape and indexing.
    data:
        Flat Tensor of length mesh.n_cells holding the cell total values.
    """

    def __init__(self, mesh: Mesh, data: Tensor) -> None:
        self._mesh = mesh
        self._data = data

    @property
    def mesh(self) -> Mesh:
        return self._mesh

    @property
    def data(self) -> Tensor:
        """The underlying flat Tensor of cell values."""
        return self._data

    def __call__(self, idx: tuple[int, ...]) -> float:  # type: ignore[override]
        shape = self._mesh.shape
        flat = 0
        stride = 1
        for a, i in enumerate(idx):
            flat += i * stride
            stride *= shape[a]
        return self._data[flat]  # type: ignore[no-any-return]  # PythonBackend may hold sympy.Expr


__all__ = ["State"]
