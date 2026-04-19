#!/usr/bin/env python
"""Generate kernel constants blocks from their derivations.

Run from the repository root:

    python scripts/generate_kernels.py

Each kernel module contains a derivation section (_derive, generate) and a
generated constants block delimited by BEGIN GENERATED / END GENERATED
sentinels. This script runs the derivation and splices the fresh constants
back into the file between those sentinels.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parent.parent

_BEGIN = (
    "# BEGIN GENERATED — do not edit; regenerate with scripts/generate_kernels.py\n"
)
_END = "# END GENERATED\n"


def splice(path: Path, block: str) -> None:
    """Replace the generated constants block in *path* with *block*."""
    source = path.read_text()
    start = source.index(_BEGIN) + len(_BEGIN)
    end = source.index(_END)
    path.write_text(source[:start] + block + source[end:])


def main() -> None:
    from cosmic_foundry.computation.stencil import generate

    target = ROOT / "cosmic_foundry" / "computation" / "stencil.py"
    splice(target, generate())
    print(f"wrote {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
