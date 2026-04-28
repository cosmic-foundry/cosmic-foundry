"""Operator: physics-layer discrete operator instantiated on a Mesh."""

from __future__ import annotations

from typing import Any

import sympy

from cosmic_foundry.computation.tensor import Tensor
from cosmic_foundry.physics.state import State
from cosmic_foundry.theory.discrete.discrete_field import _CallableDiscreteField
from cosmic_foundry.theory.discrete.discrete_operator import DiscreteOperator
from cosmic_foundry.theory.discrete.mesh import Mesh


def _to_multi(flat: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    idx = []
    for n in shape:
        idx.append(flat % n)
        flat //= n
    return tuple(idx)


class Operator:
    """A symbolic DiscreteOperator instantiated on a specific Mesh.

    Operator binds an abstract scheme (a DiscreteOperator produced by a
    Discretization) to concrete geometry (a Mesh).  At construction time it
    probes the operator symbolically — applying it once to a field of sympy
    symbols — to extract the sparse stiffness structure (row_indices,
    col_indices, stiffness_values).  This precomputation is O(N · stencil_width)
    and is done once; subsequent apply() calls are O(nnz) sparse matrix-vector
    products.

        op = Operator(disc(), mesh)
        residual  = op(state)       # symbolic path: O(N) stencil evaluation
        u         = solver.solve(op, b)  # solver uses op.apply() internally

    apply() is the primary numeric path; __call__ is the symbolic path used
    by time integration and convergence testing.

    Parameters
    ----------
    op:
        Symbolic DiscreteOperator describing the scheme.
    mesh:
        The Mesh on which the operator is instantiated.
    """

    def __init__(self, op: DiscreteOperator[Any], mesh: Mesh) -> None:
        self._op = op
        self._mesh = mesh
        self._rows, self._cols, self._vals = self._build_stiffness()

    def _build_stiffness(self) -> tuple[list[int], list[int], list[float]]:
        """Probe the operator symbolically; extract nonzero stiffness entries."""
        n = self._mesh.n_cells
        shape = self._mesh.shape
        u_syms = [sympy.Symbol(f"_u{j}") for j in range(n)]

        def _to_flat(idx: tuple[int, ...]) -> int:
            flat, stride = 0, 1
            for a, i in enumerate(idx):
                flat += i * stride
                stride *= shape[a]
            return flat

        sym_field: _CallableDiscreteField[sympy.Expr] = _CallableDiscreteField(
            self._mesh, lambda idx: u_syms[_to_flat(idx)]
        )
        result = self._op(sym_field)

        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []
        for i in range(n):
            expr = result(_to_multi(i, shape))  # type: ignore[arg-type]
            for j, sym in enumerate(u_syms):
                c = float(expr.coeff(sym))
                if c != 0.0:
                    rows.append(i)
                    cols.append(j)
                    vals.append(c)
        return rows, cols, vals

    @property
    def mesh(self) -> Mesh:
        return self._mesh

    def apply(self, u: Tensor) -> Tensor:
        """Apply the operator to u via sparse matrix-vector product; O(nnz).

        JIT-compilable on all backends: the stiffness structure (rows, cols,
        vals) is static and captured at construction time.
        """
        backend = u.backend
        n = u.shape[0]
        raw = backend.spmv(self._rows, self._cols, self._vals, u._value, n)
        return Tensor(raw, backend=backend)

    def diagonal(self, backend: Any) -> Tensor:
        """Return the diagonal entries of the stiffness matrix."""
        n = self._mesh.n_cells
        d = [0.0] * n
        for r, c, v in zip(self._rows, self._cols, self._vals, strict=True):
            if r == c:
                d[r] += v
        return Tensor(d, backend=backend)

    def row_abs_sums(self, backend: Any) -> Tensor:
        """Return per-row sums of |A_{ij}| (used for Gershgorin bound)."""
        n = self._mesh.n_cells
        s = [0.0] * n
        for r, v in zip(self._rows, self._vals, strict=True):
            s[r] += abs(v)
        return Tensor(s, backend=backend)

    def __call__(self, state: State) -> State:
        """Apply the operator to a State; return a new State of residuals.

        Evaluates the stencil cell-by-cell via the symbolic DiscreteOperator.
        """
        result = self._op(state)
        shape = self._mesh.shape
        n_total = self._mesh.n_cells
        values = [
            float(result(_to_multi(i, shape)))  # type: ignore[arg-type]
            for i in range(n_total)
        ]
        return State(self._mesh, Tensor(values))


__all__ = ["Operator"]
