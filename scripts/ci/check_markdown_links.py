#!/usr/bin/env python3
"""Verify relative markdown links resolve to existing files.

Also validates own-repo GitHub URLs (``https://github.com/cosmic-foundry/
cosmic-foundry/{blob,tree,raw}/main/…``) against the working tree. These
links appear in docs that live inside ``docs/`` but point at files outside
the docs tree (ADRs, CONTRIBUTING.md, AI.md, …). Letting ``sphinx-build
-b linkcheck`` validate them over HTTP would 404 pre-merge whenever a PR
adds both the target file and the link to it in the same change. Validating
here — against the tree that is about to become ``main`` — avoids that
bootstrap problem and still catches path typos and stale links.

Other external URLs (http, https, mailto) and pure in-page fragments
(#anchor) are skipped. Fragments on relative links (path#section) are
checked for file existence only; anchor targets are not validated.

Run from any directory; resolves the repo root from this file's location.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

LINK_RE = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")
INLINE_CODE_RE = re.compile(r"`[^`]*`")
OWN_REPO_MAIN_RE = re.compile(
    r"^https://github\.com/cosmic-foundry/cosmic-foundry/"
    r"(?:blob|tree|raw)/main/(.+)$"
)
SKIP_PREFIXES = ("http://", "https://", "mailto:", "#")
SKIP_DIRS = {"miniforge", ".git"}


def iter_md(repo_root: Path):
    for path in repo_root.rglob("*.md"):
        if SKIP_DIRS.intersection(path.parts):
            continue
        yield path


def scan(repo_root: Path) -> list[str]:
    errors: list[str] = []
    for md in iter_md(repo_root):
        in_fence = False
        for lineno, line in enumerate(md.read_text(encoding="utf-8").splitlines(), 1):
            if line.lstrip().startswith("```"):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            stripped = INLINE_CODE_RE.sub("", line)
            for match in LINK_RE.finditer(stripped):
                href = match.group(2).strip()
                if not href:
                    continue
                own_repo = OWN_REPO_MAIN_RE.match(href)
                if own_repo:
                    path = own_repo.group(1).split("#", 1)[0]
                    if not path:
                        continue
                    resolved = (repo_root / path).resolve()
                    if not resolved.exists():
                        rel = md.relative_to(repo_root)
                        errors.append(f"{rel}:{lineno}: broken own-repo link -> {href}")
                    continue
                if href.startswith(SKIP_PREFIXES):
                    continue
                target = href.split("#", 1)[0]
                if not target:
                    continue
                resolved = (md.parent / target).resolve()
                if not resolved.exists():
                    rel = md.relative_to(repo_root)
                    errors.append(f"{rel}:{lineno}: broken link -> {href}")
    return errors


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    errors = scan(repo_root)
    for err in errors:
        print(err, file=sys.stderr)
    if errors:
        print(f"\n{len(errors)} broken link(s)", file=sys.stderr)
        return 1
    print("All relative markdown links resolve.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
