"""Drift check: generated kernel files must match fresh derivation output.

If this test fails, the committed kernel file has drifted from its derivation.
Regenerate with: python scripts/generate_kernels.py
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parent.parent


def test_laplacian_kernel_matches_derivation() -> None:
    """cosmic_foundry/computation/laplacian.py must be the current output of
    derivations/laplacian_stencil.generate_kernel_source().

    Importing the derivation runs its SymPy assertions as a side-effect,
    so this test simultaneously verifies the Taylor-expansion algebra and
    the committed kernel file.
    """
    from derivations.laplacian_stencil import generate_kernel_source

    committed = (ROOT / "cosmic_foundry" / "computation" / "laplacian.py").read_text()
    fresh = generate_kernel_source()
    assert committed == fresh, (
        "cosmic_foundry/computation/laplacian.py is out of sync with the derivation. "
        "Regenerate with: python scripts/generate_kernels.py"
    )
