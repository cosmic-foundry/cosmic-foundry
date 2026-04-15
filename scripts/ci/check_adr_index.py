#!/usr/bin/env python3
"""Verify adr/README.md links every ADR file and only existing ADR files.

Reserved ADR numbers (e.g. ADR-0001 through ADR-0005 per
roadmap/epoch-00-bootstrap.md) are exempt from the "must be linked"
rule if no corresponding file exists yet.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ADR_FILE_RE = re.compile(r"^ADR-\d{4}-.*\.md$")
LINK_RE = re.compile(r"\(([^)]*ADR-\d{4}[^)]*\.md)\)")


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    adr_dir = repo_root / "adr"
    index = adr_dir / "README.md"

    if not index.exists():
        print("adr/README.md: missing", file=sys.stderr)
        return 1

    adr_files = {p.name for p in adr_dir.iterdir() if ADR_FILE_RE.match(p.name)}
    index_text = index.read_text(encoding="utf-8")
    linked = {Path(m).name for m in LINK_RE.findall(index_text)}

    errors: list[str] = []
    for name in sorted(adr_files - linked):
        errors.append(f"adr/{name}: exists but is not linked from adr/README.md")
    for name in sorted(linked - adr_files):
        errors.append(f"adr/README.md: links to missing {name}")

    for err in errors:
        print(err, file=sys.stderr)
    if errors:
        return 1
    print(f"ADR index consistent ({len(adr_files)} ADR(s) listed).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
