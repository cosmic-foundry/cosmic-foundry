#!/usr/bin/env python
"""Generate kernel source files from their derivations.

Run from the repository root:

    python scripts/generate_kernels.py

Each generated file embeds a hash of its derived coefficients so that
import-time integrity checks can detect hand-edits without running SymPy.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parent.parent


def main() -> None:
    from derivations.laplacian_stencil import generate_kernel_source

    target = ROOT / "cosmic_foundry" / "computation" / "laplacian.py"
    source = generate_kernel_source()
    target.write_text(source)
    print(f"wrote {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
