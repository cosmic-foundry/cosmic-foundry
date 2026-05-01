#!/usr/bin/env python3
"""Enforce that cosmic_foundry/computation/ does not import from other
cosmic_foundry sub-packages.

Files inside computation/ may only import from:
  - the standard library and third-party packages (numpy, math, …)
  - cosmic_foundry.computation itself

Importing from cosmic_foundry.geometry, cosmic_foundry.theory, or any
other cosmic_foundry sub-package is a layering violation: computation/
is the bottom-most numeric layer and must remain mesh-agnostic.

Uses the AST to detect imports, so strings and comments are ignored.
"""
from __future__ import annotations

import sys

from import_dependencies import computation_layer_violations


def main() -> int:
    errors = computation_layer_violations()
    for err in errors:
        print(err, file=sys.stderr)
    if errors:
        print(
            f"\n{len(errors)} violation(s) — computation/ must not import from other"
            " cosmic_foundry packages",
            file=sys.stderr,
        )
        return 1
    print("No layering violations in cosmic_foundry/computation/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
