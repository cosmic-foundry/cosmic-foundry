#!/usr/bin/env python3
"""Enforce that numeric computation libraries are only imported inside
cosmic_foundry/computation/.

The libraries math, numpy, scipy, jax, and torch must not be imported
anywhere in cosmic_foundry/ outside the computation sub-package.  All
numeric operations on Tensor data belong in that layer; higher-level
layers (theory, geometry, …) interact with numbers exclusively through
the Tensor API.

Uses the AST to detect imports, so strings and comments are ignored.
"""
from __future__ import annotations

import sys

from import_dependencies import numeric_import_violations


def main() -> int:
    errors = numeric_import_violations()
    for err in errors:
        print(err, file=sys.stderr)
    if errors:
        print(
            f"\n{len(errors)} violation(s) — numeric imports outside computation/",
            file=sys.stderr,
        )
        return 1
    print("No numeric imports outside cosmic_foundry/computation/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
