#!/usr/bin/env python3
"""Select pytest targets from changed files using the static import graph."""
from __future__ import annotations

import argparse

from import_dependencies import changed_files, selected_pytest_targets


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="origin/main")
    parser.add_argument("--head", default="HEAD")
    args = parser.parse_args()
    print(" ".join(selected_pytest_targets(changed_files(args.base, args.head))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
