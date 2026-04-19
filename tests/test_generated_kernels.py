"""Drift check: generated constants blocks must match fresh derivation output.

If this test fails, the constants in the kernel file have drifted from the
derivation. Regenerate with: python scripts/generate_kernels.py
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parent.parent

_BEGIN = (
    "# BEGIN GENERATED — do not edit; regenerate with scripts/generate_kernels.py\n"
)
_END = "# END GENERATED\n"


def _extract_generated_block(path: Path) -> str:
    source = path.read_text()
    start = source.index(_BEGIN) + len(_BEGIN)
    end = source.index(_END)
    return source[start:end]


def test_laplacian_constants_match_derivation() -> None:
    """The generated constants block in stencil.py must match a fresh
    derivation run. Calling generate() runs the SymPy Taylor-expansion
    assertions as a side-effect, so this test simultaneously verifies the
    algebra and the committed constants.
    """
    from cosmic_foundry.computation.stencil import generate

    committed = _extract_generated_block(
        ROOT / "cosmic_foundry" / "computation" / "stencil.py"
    )
    assert committed == generate(), (
        "cosmic_foundry/computation/stencil.py constants are out of sync "
        "with the derivation. Regenerate with: python scripts/generate_kernels.py"
    )
