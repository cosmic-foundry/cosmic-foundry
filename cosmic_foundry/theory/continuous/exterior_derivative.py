"""ExteriorDerivative: d: Ω^k → Ω^{k+1}."""

from __future__ import annotations

from typing import Any

import sympy

from cosmic_foundry.theory.continuous.differential_form import (
    DifferentialForm,
    OneForm,
    ThreeForm,
    TwoForm,
    ZeroForm,
)
from cosmic_foundry.theory.continuous.differential_operator import DifferentialOperator
from cosmic_foundry.theory.continuous.manifold import Manifold


class ExteriorDerivative(DifferentialOperator[Any, Any]):
    """The exterior derivative d: Ω^k → Ω^{k+1}.

    Concrete for degrees 0, 1, 2:
        d₀: ZeroForm  → OneForm   (gradient: (df)ᵢ = ∂f/∂xᵢ)
        d₁: OneForm   → TwoForm   (curl: (dω)ᵢⱼ = ∂ωⱼ/∂xᵢ − ∂ωᵢ/∂xⱼ)
        d₂: TwoForm   → ThreeForm (in 3D: ∂F₁₂/∂x₀ − ∂F₀₂/∂x₁ + ∂F₀₁/∂x₂)

    The identity d∘d = 0 holds exactly: d₁(d₀(f)) = 0 for any ZeroForm f,
    and d₂(d₁(ω)) = 0 for any OneForm ω (follows from symmetry of mixed
    partial derivatives).

    Parameters
    ----------
    manifold:
        The manifold on which the operator acts.
    degree:
        Input form degree k ∈ {0, 1, 2}.
    """

    def __init__(self, manifold: Manifold, degree: int) -> None:
        if degree not in (0, 1, 2):
            raise ValueError(
                f"ExteriorDerivative degree must be 0, 1, or 2; got {degree}"
            )
        self._manifold = manifold
        self._degree = degree

    @property
    def manifold(self) -> Manifold:
        return self._manifold

    @property
    def order(self) -> int:
        return 1

    @property
    def degree(self) -> int:
        """Input form degree k."""
        return self._degree

    def __call__(self, f: DifferentialForm) -> DifferentialForm:
        if self._degree == 0:
            return self._d0(f)  # type: ignore[arg-type]
        if self._degree == 1:
            return self._d1(f)  # type: ignore[arg-type]
        return self._d2(f)  # type: ignore[arg-type]

    def _d0(self, f: ZeroForm) -> OneForm:
        components = tuple(sympy.diff(f.expr, s) for s in f.symbols)
        return OneForm(f.manifold, components, f.symbols)

    def _d1(self, f: OneForm) -> TwoForm:
        n = len(f.symbols)
        matrix = sympy.zeros(n, n)
        for i in range(n):
            for j in range(i + 1, n):
                val = sympy.diff(f.component(j), f.symbols[i]) - sympy.diff(
                    f.component(i), f.symbols[j]
                )
                matrix[i, j] = val
                matrix[j, i] = -val
        return TwoForm(f.manifold, matrix, f.symbols)

    def _d2(self, f: TwoForm) -> ThreeForm:
        n = len(f.symbols)
        if n != 3:
            raise NotImplementedError(
                f"ExteriorDerivative d₂ is only implemented for n=3; got n={n}"
            )
        # In 3D: scalar = ∂F₁₂/∂x₀ − ∂F₀₂/∂x₁ + ∂F₀₁/∂x₂
        expr = (
            sympy.diff(f.component(1, 2), f.symbols[0])
            - sympy.diff(f.component(0, 2), f.symbols[1])
            + sympy.diff(f.component(0, 1), f.symbols[2])
        )
        return ThreeForm(f.manifold, expr, f.symbols)


__all__ = ["ExteriorDerivative"]
